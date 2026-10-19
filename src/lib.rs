//! Typed, protobuf-backed key/value storage shared across projects.
//!
//! Repositories depend on [`Table`]. Production uses [`RedbTable`], while
//! tests use [`MemoryTable`] through the same protobuf serialization path.

// The initial release preserves HuntersLife's public DataStoreError variants.
// Boxing the large redb transaction variant would break source compatibility.
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};

use prost::Message;
use redb::{Database, ReadableTable, TableDefinition};

#[derive(thiserror::Error, Debug)]
pub enum DataStoreError {
    #[error("redb database error: {0}")]
    Database(#[from] redb::DatabaseError),
    #[error("redb transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),
    #[error("redb table error: {0}")]
    Table(#[from] redb::TableError),
    #[error("redb storage error: {0}")]
    Storage(#[from] redb::StorageError),
    #[error("redb commit error: {0}")]
    Commit(#[from] redb::CommitError),
    #[error("protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error("protobuf encode error: {0}")]
    Encode(#[from] prost::EncodeError),
    #[error("key already exists in table {table}")]
    AlreadyExists { table: &'static str },
    #[error("value precondition failed in table {table}")]
    PreconditionFailed { table: &'static str },
}

/// Marker trait for protobuf message types that may be stored as values.
pub trait DataType: Default + Clone + Send + Sync + Sized + PartialEq + Message + 'static {
    fn parse_from_bytes(buffer: &[u8]) -> Result<Self, DataStoreError> {
        Self::decode(buffer).map_err(DataStoreError::Decode)
    }

    fn to_bytes(&self) -> Result<Vec<u8>, DataStoreError> {
        let mut buffer = Vec::with_capacity(self.encoded_len());
        self.encode(&mut buffer).map_err(DataStoreError::Encode)?;
        Ok(buffer)
    }
}

impl<T> DataType for T where T: Default + Clone + Send + Sync + Sized + PartialEq + Message + 'static
{}

/// Storage abstraction implemented by persistent and in-memory backends.
pub trait Table<D: DataType>: Send + Sync {
    fn look_up(&self, key: &[u8]) -> Result<Option<D>, DataStoreError>;
    fn insert(&self, key: &[u8], value: &D) -> Result<(), DataStoreError>;
    fn remove(&self, key: &[u8]) -> Result<bool, DataStoreError>;
    fn scan_all(&self) -> Result<Vec<(Vec<u8>, D)>, DataStoreError>;
    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, D)>, DataStoreError>;
}

/// One encoded mutation applied by [`AtomicWriteStore`].
enum WriteOperation {
    Put {
        table: &'static str,
        key: Vec<u8>,
        value: Vec<u8>,
        require_absent: bool,
        expected_value: Option<Vec<u8>>,
    },
    Remove {
        table: &'static str,
        key: Vec<u8>,
    },
}

/// Type-safe builder for mutations committed as one storage transaction.
#[derive(Default)]
pub struct WriteBatch {
    operations: Vec<WriteOperation>,
}

impl WriteBatch {
    /// Creates an empty batch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Encodes and appends one typed insert or replacement.
    ///
    /// Encoding happens before a backend transaction begins, so encoding
    /// failure cannot leave a partially applied batch.
    ///
    /// # Errors
    ///
    /// Returns [`DataStoreError::Encode`] when `value` cannot be encoded.
    pub fn insert<D: DataType>(
        &mut self,
        table: &'static str,
        key: impl Into<Vec<u8>>,
        value: &D,
    ) -> Result<(), DataStoreError> {
        self.operations.push(WriteOperation::Put {
            table,
            key: key.into(),
            value: value.to_bytes()?,
            require_absent: false,
            expected_value: None,
        });
        Ok(())
    }

    /// Encodes and appends an insert that fails if the key already exists.
    ///
    /// # Errors
    ///
    /// Returns [`DataStoreError::Encode`] when `value` cannot be encoded.
    /// [`AtomicWriteStore::apply_batch`] returns
    /// [`DataStoreError::AlreadyExists`] if the key exists at commit time.
    pub fn insert_if_absent<D: DataType>(
        &mut self,
        table: &'static str,
        key: impl Into<Vec<u8>>,
        value: &D,
    ) -> Result<(), DataStoreError> {
        self.operations.push(WriteOperation::Put {
            table,
            key: key.into(),
            value: value.to_bytes()?,
            require_absent: true,
            expected_value: None,
        });
        Ok(())
    }

    /// Encodes and appends a replacement that requires an exact current value.
    ///
    /// Both values are encoded before a backend transaction begins. The
    /// backend replaces the value only when the current encoded bytes exactly
    /// equal `expected_value`.
    ///
    /// # Errors
    ///
    /// Returns [`DataStoreError::Encode`] when either value cannot be encoded.
    /// [`AtomicWriteStore::apply_batch`] returns
    /// [`DataStoreError::PreconditionFailed`] if the current value differs or
    /// the key does not exist at commit time.
    pub fn replace_if_equal<D: DataType>(
        &mut self,
        table: &'static str,
        key: impl Into<Vec<u8>>,
        expected_value: &D,
        new_value: &D,
    ) -> Result<(), DataStoreError> {
        self.operations.push(WriteOperation::Put {
            table,
            key: key.into(),
            value: new_value.to_bytes()?,
            require_absent: false,
            expected_value: Some(expected_value.to_bytes()?),
        });
        Ok(())
    }

    /// Appends one key removal to this batch.
    pub fn remove(&mut self, table: &'static str, key: impl Into<Vec<u8>>) {
        self.operations.push(WriteOperation::Remove {
            table,
            key: key.into(),
        });
    }

    /// Returns whether this batch contains no mutations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Returns the number of mutations in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

/// Backend capable of applying an encoded batch atomically.
pub trait AtomicWriteStore: Send + Sync {
    /// Applies every operation or leaves the backend unchanged.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the transaction cannot be opened,
    /// mutated, or committed.
    fn apply_batch(&self, batch: WriteBatch) -> Result<(), DataStoreError>;
}

/// Handle to a `redb` database shared by typed tables.
pub struct Store {
    db: Database,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, DataStoreError> {
        let db = Database::create(path)?;
        Ok(Arc::new(Self { db }))
    }
}

impl AtomicWriteStore for Store {
    fn apply_batch(&self, batch: WriteBatch) -> Result<(), DataStoreError> {
        if batch.is_empty() {
            return Ok(());
        }

        let write = self.db.begin_write()?;
        for operation in batch.operations {
            match operation {
                WriteOperation::Put {
                    table,
                    key,
                    value,
                    require_absent,
                    expected_value,
                } => {
                    let table_name = table;
                    let definition: TableDefinition<&[u8], &[u8]> =
                        TableDefinition::new(table_name);
                    let mut table_handle = write.open_table(definition)?;
                    if require_absent && table_handle.get(key.as_slice())?.is_some() {
                        return Err(DataStoreError::AlreadyExists { table: table_name });
                    }
                    if let Some(expected_value) = expected_value {
                        let current_value = table_handle
                            .get(key.as_slice())?
                            .map(|guard| guard.value().to_vec());
                        if current_value.as_deref() != Some(expected_value.as_slice()) {
                            return Err(DataStoreError::PreconditionFailed { table: table_name });
                        }
                    }
                    table_handle.insert(key.as_slice(), value.as_slice())?;
                }
                WriteOperation::Remove { table, key } => {
                    let definition: TableDefinition<&[u8], &[u8]> = TableDefinition::new(table);
                    let mut table = write.open_table(definition)?;
                    table.remove(key.as_slice())?;
                }
            }
        }
        write.commit()?;
        Ok(())
    }
}

/// Typed view over a `redb` table.
pub struct RedbTable<D: DataType> {
    store: Arc<Store>,
    definition: TableDefinition<'static, &'static [u8], &'static [u8]>,
    marker: PhantomData<D>,
}

impl<D: DataType> RedbTable<D> {
    pub fn open(store: Arc<Store>, name: &'static str) -> Result<Self, DataStoreError> {
        let definition = TableDefinition::new(name);
        let write = store.db.begin_write()?;
        {
            let _table = write.open_table(definition)?;
        }
        write.commit()?;
        Ok(Self {
            store,
            definition,
            marker: PhantomData,
        })
    }
}

impl<D: DataType> Table<D> for RedbTable<D> {
    fn look_up(&self, key: &[u8]) -> Result<Option<D>, DataStoreError> {
        let read = self.store.db.begin_read()?;
        let table = read.open_table(self.definition)?;
        match table.get(key)? {
            Some(guard) => Ok(Some(D::parse_from_bytes(guard.value())?)),
            None => Ok(None),
        }
    }

    fn insert(&self, key: &[u8], value: &D) -> Result<(), DataStoreError> {
        let bytes = value.to_bytes()?;
        let write = self.store.db.begin_write()?;
        {
            let mut table = write.open_table(self.definition)?;
            table.insert(key, bytes.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }

    fn remove(&self, key: &[u8]) -> Result<bool, DataStoreError> {
        let write = self.store.db.begin_write()?;
        let existed;
        {
            let mut table = write.open_table(self.definition)?;
            existed = table.remove(key)?.is_some();
        }
        write.commit()?;
        Ok(existed)
    }

    fn scan_all(&self) -> Result<Vec<(Vec<u8>, D)>, DataStoreError> {
        let read = self.store.db.begin_read()?;
        let table = read.open_table(self.definition)?;
        let mut rows = Vec::new();
        for entry in table.iter()? {
            let (key, value) = entry?;
            rows.push((key.value().to_vec(), D::parse_from_bytes(value.value())?));
        }
        Ok(rows)
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, D)>, DataStoreError> {
        let read = self.store.db.begin_read()?;
        let table = read.open_table(self.definition)?;
        let mut rows = Vec::new();
        for entry in table.range::<&[u8]>(prefix..)? {
            let (key, value) = entry?;
            let key = key.value();
            if !key.starts_with(prefix) {
                break;
            }
            rows.push((key.to_vec(), D::parse_from_bytes(value.value())?));
        }
        Ok(rows)
    }
}

/// In-memory table that preserves the production protobuf serialization path.
pub struct MemoryTable<D: DataType> {
    inner: Mutex<BTreeMap<Vec<u8>, Vec<u8>>>,
    marker: PhantomData<D>,
}

impl<D: DataType> MemoryTable<D> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BTreeMap::new()),
            marker: PhantomData,
        }
    }
}

impl<D: DataType> Default for MemoryTable<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: DataType> Table<D> for MemoryTable<D> {
    fn look_up(&self, key: &[u8]) -> Result<Option<D>, DataStoreError> {
        let rows = self.inner.lock().expect("MemoryTable lock poisoned");
        match rows.get(key) {
            Some(bytes) => Ok(Some(D::parse_from_bytes(bytes)?)),
            None => Ok(None),
        }
    }

    fn insert(&self, key: &[u8], value: &D) -> Result<(), DataStoreError> {
        self.inner
            .lock()
            .expect("MemoryTable lock poisoned")
            .insert(key.to_vec(), value.to_bytes()?);
        Ok(())
    }

    fn remove(&self, key: &[u8]) -> Result<bool, DataStoreError> {
        Ok(self
            .inner
            .lock()
            .expect("MemoryTable lock poisoned")
            .remove(key)
            .is_some())
    }

    fn scan_all(&self) -> Result<Vec<(Vec<u8>, D)>, DataStoreError> {
        let rows = self.inner.lock().expect("MemoryTable lock poisoned");
        rows.iter()
            .map(|(key, value)| Ok((key.clone(), D::parse_from_bytes(value)?)))
            .collect()
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, D)>, DataStoreError> {
        let rows = self.inner.lock().expect("MemoryTable lock poisoned");
        let mut matches = Vec::new();
        for (key, value) in rows.range(prefix.to_vec()..) {
            if !key.starts_with(prefix) {
                break;
            }
            matches.push((key.clone(), D::parse_from_bytes(value)?));
        }
        Ok(matches)
    }
}

type EncodedRows = BTreeMap<Vec<u8>, Vec<u8>>;
type EncodedTables = BTreeMap<&'static str, EncodedRows>;

/// Shared in-memory backend supporting typed tables and atomic batches.
#[derive(Clone, Default)]
pub struct MemoryStore {
    inner: Arc<Mutex<EncodedTables>>,
}

impl MemoryStore {
    /// Creates an empty shared in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens a typed table view sharing this store's atomic backing state.
    pub fn table<D: DataType>(&self, name: &'static str) -> MemoryStoreTable<D> {
        MemoryStoreTable {
            store: self.clone(),
            name,
            marker: PhantomData,
        }
    }
}

impl AtomicWriteStore for MemoryStore {
    fn apply_batch(&self, batch: WriteBatch) -> Result<(), DataStoreError> {
        let mut tables = self.inner.lock().expect("MemoryStore lock poisoned");
        let mut candidate = tables.clone();
        for operation in batch.operations {
            match operation {
                WriteOperation::Put {
                    table,
                    key,
                    value,
                    require_absent,
                    expected_value,
                } => {
                    let rows = candidate.entry(table).or_default();
                    if require_absent && rows.contains_key(&key) {
                        return Err(DataStoreError::AlreadyExists { table });
                    }
                    if let Some(expected_value) = expected_value
                        && rows.get(&key) != Some(&expected_value)
                    {
                        return Err(DataStoreError::PreconditionFailed { table });
                    }
                    rows.insert(key, value);
                }
                WriteOperation::Remove { table, key } => {
                    if let Some(rows) = candidate.get_mut(table) {
                        rows.remove(&key);
                    }
                }
            }
        }
        *tables = candidate;
        Ok(())
    }
}

/// Typed table view backed by a shared [`MemoryStore`].
pub struct MemoryStoreTable<D: DataType> {
    store: MemoryStore,
    name: &'static str,
    marker: PhantomData<D>,
}

impl<D: DataType> Table<D> for MemoryStoreTable<D> {
    fn look_up(&self, key: &[u8]) -> Result<Option<D>, DataStoreError> {
        let tables = self.store.inner.lock().expect("MemoryStore lock poisoned");
        tables
            .get(self.name)
            .and_then(|rows| rows.get(key))
            .map(|value| D::parse_from_bytes(value))
            .transpose()
    }

    fn insert(&self, key: &[u8], value: &D) -> Result<(), DataStoreError> {
        let mut batch = WriteBatch::new();
        batch.insert(self.name, key.to_vec(), value)?;
        self.store.apply_batch(batch)
    }

    fn remove(&self, key: &[u8]) -> Result<bool, DataStoreError> {
        let existed = self.look_up(key)?.is_some();
        let mut batch = WriteBatch::new();
        batch.remove(self.name, key.to_vec());
        self.store.apply_batch(batch)?;
        Ok(existed)
    }

    fn scan_all(&self) -> Result<Vec<(Vec<u8>, D)>, DataStoreError> {
        self.scan_prefix(&[])
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, D)>, DataStoreError> {
        let tables = self.store.inner.lock().expect("MemoryStore lock poisoned");
        let Some(rows) = tables.get(self.name) else {
            return Ok(Vec::new());
        };
        let mut matches = Vec::new();
        for (key, value) in rows.range(prefix.to_vec()..) {
            if !key.starts_with(prefix) {
                break;
            }
            matches.push((key.clone(), D::parse_from_bytes(value)?));
        }
        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, ::prost::Message)]
    struct Blob {
        #[prost(bytes = "vec", tag = "1")]
        payload: Vec<u8>,
    }

    fn blob(payload: &[u8]) -> Blob {
        Blob {
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn scan_prefix_returns_only_matching_keys() {
        let table = MemoryTable::<Blob>::new();
        table.insert(b"user-A/item-1", &blob(b"a1")).unwrap();
        table.insert(b"user-A/item-2", &blob(b"a2")).unwrap();
        table.insert(b"user-B/item-1", &blob(b"b1")).unwrap();

        let rows = table.scan_prefix(b"user-A/").unwrap();
        let keys: Vec<&[u8]> = rows.iter().map(|(key, _)| key.as_slice()).collect();
        assert_eq!(keys, vec![&b"user-A/item-1"[..], &b"user-A/item-2"[..]]);
    }

    #[test]
    fn scan_prefix_returns_in_sorted_order() {
        let table = MemoryTable::<Blob>::new();
        table.insert(b"u/03", &blob(b"3")).unwrap();
        table.insert(b"u/01", &blob(b"1")).unwrap();
        table.insert(b"u/02", &blob(b"2")).unwrap();

        let rows = table.scan_prefix(b"u/").unwrap();
        let keys: Vec<&[u8]> = rows.iter().map(|(key, _)| key.as_slice()).collect();
        assert_eq!(keys, vec![&b"u/01"[..], &b"u/02"[..], &b"u/03"[..]]);
    }

    #[test]
    fn empty_prefix_returns_all_rows() {
        let table = MemoryTable::<Blob>::new();
        table.insert(b"a", &blob(b"1")).unwrap();
        table.insert(b"b", &blob(b"2")).unwrap();
        assert_eq!(table.scan_prefix(b"").unwrap().len(), 2);
    }

    #[test]
    fn atomic_batch_updates_multiple_typed_tables() {
        let store = MemoryStore::new();
        let first = store.table::<Blob>("first");
        let second = store.table::<Blob>("second");
        let mut batch = WriteBatch::new();
        batch.insert("first", b"a".to_vec(), &blob(b"1")).unwrap();
        batch.insert("second", b"b".to_vec(), &blob(b"2")).unwrap();

        store.apply_batch(batch).unwrap();

        assert_eq!(first.look_up(b"a").unwrap(), Some(blob(b"1")));
        assert_eq!(second.look_up(b"b").unwrap(), Some(blob(b"2")));
    }

    #[test]
    fn failed_conditional_batch_leaves_every_table_unchanged() {
        let store = MemoryStore::new();
        let first = store.table::<Blob>("first");
        let second = store.table::<Blob>("second");
        first.insert(b"existing", &blob(b"old")).unwrap();
        let mut batch = WriteBatch::new();
        batch
            .insert_if_absent("second", b"new".to_vec(), &blob(b"new"))
            .unwrap();
        batch
            .insert_if_absent("first", b"existing".to_vec(), &blob(b"replace"))
            .unwrap();

        assert!(matches!(
            store.apply_batch(batch),
            Err(DataStoreError::AlreadyExists { table: "first" })
        ));
        assert_eq!(first.look_up(b"existing").unwrap(), Some(blob(b"old")));
        assert_eq!(second.look_up(b"new").unwrap(), None);
    }

    #[test]
    fn redb_conditional_conflict_rolls_back_prior_cross_table_write() {
        let directory = tempfile::tempdir().unwrap();
        let store = Store::open(directory.path().join("atomic.redb")).unwrap();
        let first = RedbTable::<Blob>::open(store.clone(), "first").unwrap();
        let second = RedbTable::<Blob>::open(store.clone(), "second").unwrap();
        first.insert(b"existing", &blob(b"old")).unwrap();
        let mut batch = WriteBatch::new();
        batch
            .insert_if_absent("second", b"new".to_vec(), &blob(b"new"))
            .unwrap();
        batch
            .insert_if_absent("first", b"existing".to_vec(), &blob(b"replace"))
            .unwrap();

        assert!(matches!(
            store.apply_batch(batch),
            Err(DataStoreError::AlreadyExists { table: "first" })
        ));
        assert_eq!(first.look_up(b"existing").unwrap(), Some(blob(b"old")));
        assert_eq!(second.look_up(b"new").unwrap(), None);
    }

    #[test]
    fn memory_compare_and_swap_replaces_only_expected_value() {
        let store = MemoryStore::new();
        let table = store.table::<Blob>("records");
        table.insert(b"key", &blob(b"v1")).unwrap();

        let mut successful = WriteBatch::new();
        successful
            .replace_if_equal("records", b"key".to_vec(), &blob(b"v1"), &blob(b"v2"))
            .unwrap();
        store.apply_batch(successful).unwrap();
        assert_eq!(table.look_up(b"key").unwrap(), Some(blob(b"v2")));

        let mut stale = WriteBatch::new();
        stale
            .replace_if_equal("records", b"key".to_vec(), &blob(b"v1"), &blob(b"v3"))
            .unwrap();
        assert!(matches!(
            store.apply_batch(stale),
            Err(DataStoreError::PreconditionFailed { table: "records" })
        ));
        assert_eq!(table.look_up(b"key").unwrap(), Some(blob(b"v2")));
    }

    #[test]
    fn redb_compare_and_swap_replaces_only_expected_value() {
        let directory = tempfile::tempdir().unwrap();
        let store = Store::open(directory.path().join("cas.redb")).unwrap();
        let table = RedbTable::<Blob>::open(store.clone(), "records").unwrap();
        table.insert(b"key", &blob(b"v1")).unwrap();

        let mut successful = WriteBatch::new();
        successful
            .replace_if_equal("records", b"key".to_vec(), &blob(b"v1"), &blob(b"v2"))
            .unwrap();
        store.apply_batch(successful).unwrap();

        let mut stale = WriteBatch::new();
        stale
            .replace_if_equal("records", b"key".to_vec(), &blob(b"v1"), &blob(b"v3"))
            .unwrap();
        assert!(matches!(
            store.apply_batch(stale),
            Err(DataStoreError::PreconditionFailed { table: "records" })
        ));
        assert_eq!(table.look_up(b"key").unwrap(), Some(blob(b"v2")));
    }
}
