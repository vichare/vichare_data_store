# vichare_data_store

Typed, protobuf-backed storage shared by Vichare projects.

The initial API was extracted without behavioral changes from
`vichare/HuntersLife@2a398c68d84c5fe8eeb57a8fe78bdbeebc7ca34d`.

## Backends

- `RedbTable<D>` for persistent `redb` storage.
- `MemoryTable<D>` for tests using the same protobuf encode/decode path.
- `AtomicWriteStore` + `WriteBatch` for all-or-nothing multi-table writes.
- `MemoryStore` for tests that need typed tables sharing one atomic backend.

`WriteBatch::insert_if_absent` provides create-only semantics: a conflicting
key rejects the entire batch without replacing existing rows or retaining
earlier operations from that batch.

## Compatibility Alias

Existing projects can preserve `data_store::...` imports:

```toml
data_store = { package = "vichare_data_store", git = "<repository-url>", rev = "<commit>" }
```

During local development, replace `git` and `rev` with:

```toml
data_store = { package = "vichare_data_store", path = "../vichare_data_store" }
```

Application repositories and services should depend on `Table<D>`, not on a
concrete backend. Storage stays free of application-domain types.