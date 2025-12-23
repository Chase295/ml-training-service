# 🗄️ SQL-Dateien

Dieser Ordner enthält alle SQL-Dateien für das ML Training Service Projekt.

## 📄 Verfügbare SQL-Dateien

### Schema
- **schema.sql** - Haupt-Datenbank-Schema (Tabellen, Indizes, Constraints)

### Queries
- **cloudbeaver_queries.sql** - Beispiel-Queries für CloudBeaver (Datenbank-Explorer)

## 🚀 Verwendung

### Schema anwenden
```bash
# Mit psql
psql -h localhost -U postgres -d crypto_bot -f sql/schema.sql

# Mit Docker
docker-compose exec postgres psql -U postgres -d crypto_bot -f /sql/schema.sql
```

### CloudBeaver Queries
Die Queries in `cloudbeaver_queries.sql` können direkt in CloudBeaver ausgeführt werden, um die Datenbank zu inspizieren.

## 📝 Hinweise

- Stelle sicher, dass die Datenbank existiert, bevor du das Schema anwendest
- Backup der Datenbank vor Schema-Änderungen erstellen
- Prüfe die Verbindungsdaten in `app/database/connection.py`

