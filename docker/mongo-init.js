// Runs once, when the Mongo container is first created (via mongosh).
// Creates the read-only user `talktable_ro` that the TalkTable app connects with.
const dbName = process.env.MONGO_INITDB_DATABASE || "bookstore";

db.getSiblingDB(dbName).createUser({
  user: "talktable_ro",
  pwd: process.env.TALKTABLE_RO_PASSWORD,
  roles: [{ role: "read", db: dbName }],
});
