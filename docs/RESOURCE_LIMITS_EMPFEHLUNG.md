# 💻 Resource Limits - Empfehlungen für ML Training Service

**VM-Spezifikationen:** 8GB RAM, 4 CPU-Kerne

---

## 📊 Empfohlene Einstellungen

### CPU Limits

**Number of CPUs:** `3`
- **Warum:** 3 von 4 Kernen für ML-Training
- **Grund:** Lässt 1 Kern für System/OS frei
- **Hinweis:** ML-Training (scikit-learn, XGBoost) nutzt alle verfügbaren Kerne

**CPU sets to use:** `0-2`
- **Warum:** Erste 3 Kerne verwenden
- **Grund:** Bessere Performance bei NUMA-Systemen

**CPU Weight:** `1024` (Standard)
- **Warum:** Standard-Wert ist ausreichend
- **Grund:** Priorität ist normal

---

### Memory Limits

**Soft Memory Limit:** `6144` (6GB)
- **Warum:** 6GB für Container
- **Grund:** Lässt 2GB für System/OS frei
- **Hinweis:** Feature-Engineering kann viel RAM verbrauchen!

**Swappiness:** `60` (Standard)
- **Warum:** Standard-Wert ist ok
- **Grund:** Swap wird nur bei Bedarf verwendet

**Maximum Memory Limit:** `7168` (7GB)
- **Warum:** Maximal 7GB für Container
- **Grund:** Verhindert OOM-Kills, lässt 1GB für System
- **⚠️ WICHTIG:** Feature-Engineering mit vielen Daten kann 5-6GB benötigen!

**Maximum Swap Limit:** `2048` (2GB) - Optional
- **Warum:** Zusätzlicher Swap für Notfälle
- **Grund:** Verhindert OOM-Kills bei Spitzen
- **Hinweis:** Swap ist langsamer als RAM, aber besser als Crash

---

## ⚠️ Wichtige Hinweise

### Feature-Engineering Performance

**Bei Feature-Engineering aktiviert:**
- **Viele Daten (3+ Tage):** Kann 5-6GB RAM benötigen
- **Viele Features (~40):** Kann 10-30 Minuten dauern
- **Empfehlung:** Maximum Memory Limit auf 7GB setzen

**Ohne Feature-Engineering:**
- **Weniger RAM:** 2-3GB ausreichend
- **Schneller:** 1-5 Minuten Training

### CPU-Nutzung

**ML-Training nutzt alle verfügbaren Kerne:**
- Random Forest: Parallele Tree-Erstellung
- XGBoost: Parallele Boosting-Iterationen
- **Empfehlung:** 3 Kerne für Training, 1 Kern für System

### Memory-Management

**Bei großen Datensätzen:**
- Feature-Engineering erstellt ~40 Features
- Rolling-Berechnungen über viele Zeilen
- **Beispiel:** 37,000 Zeilen × 40 Features = viele Berechnungen

---

## 🎯 Konfiguration für verschiedene Szenarien

### Szenario 1: Schnelle Tests (ohne Feature-Engineering)
```
CPU: 2 Kerne
Memory: 4GB Maximum
→ Schnelles Training, weniger Ressourcen
```

### Szenario 2: Produktion (mit Feature-Engineering)
```
CPU: 3 Kerne
Memory: 7GB Maximum
→ Beste Performance, ausreichend RAM
```

### Szenario 3: Maximale Performance (viele Daten)
```
CPU: 3 Kerne
Memory: 7GB Maximum + 2GB Swap
→ Verhindert OOM-Kills bei großen Datensätzen
```

---

## 📝 Checkliste

- [ ] CPU: 3 Kerne (von 4)
- [ ] CPU sets: 0-2
- [ ] Soft Memory: 6GB
- [ ] Maximum Memory: 7GB
- [ ] Maximum Swap: 2GB (optional, empfohlen)
- [ ] Swappiness: 60 (Standard)

---

**Erstellt:** 2025-12-24  
**Version:** 1.0

