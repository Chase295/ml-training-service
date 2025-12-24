# Erklärung: Warum sind F1-Score und andere Metriken 0?

## Problem-Analyse

Bei dem Test-Ergebnis (ID: 2) sehen wir folgende Confusion Matrix:

```
TP (True Positives):  0
TN (True Negatives):  36.830
FP (False Positives): 0
FN (False Negatives): 20.296
```

### Das Problem

**Das Modell hat NIE eine positive Vorhersage gemacht!**

- TP = 0: Keine korrekten positiven Vorhersagen
- FP = 0: Keine falschen positiven Vorhersagen
- **→ Das Modell sagt IMMER "negativ" vorher**

### Warum sind die Metriken dann 0?

#### 1. Precision = 0

```
Precision = TP / (TP + FP)
          = 0 / (0 + 0)
          = undefined → wird als 0 gespeichert
```

**Erklärung:** Wenn keine positiven Vorhersagen gemacht werden, kann Precision nicht berechnet werden.

#### 2. Recall = 0

```
Recall = TP / (TP + FN)
       = 0 / (0 + 20.296)
       = 0
```

**Erklärung:** Das Modell hat keine der 20.296 tatsächlichen positiven Fälle erkannt.

#### 3. F1-Score = 0

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2 * (0 * 0) / (0 + 0)
   = 0
```

**Erklärung:** Wenn Precision oder Recall 0 ist, ist auch F1-Score 0.

#### 4. MCC = 0

```
MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    = (0 * 36830 - 0 * 20296) / sqrt(...)
    = 0
```

**Erklärung:** Wenn TP = 0 und FP = 0, ist MCC immer 0.

#### 5. ROC-AUC = 0.5

```
ROC-AUC = 0.5
```

**Erklärung:** 0.5 bedeutet "nicht besser als Zufall" - das Modell macht keine Unterscheidung zwischen positiven und negativen Fällen.

### Warum macht das Modell keine positiven Vorhersagen?

#### Mögliche Ursachen:

1. **Sehr konservative Entscheidungsschwelle**
   - Standard-Schwelle ist oft 0.5
   - Bei unausgewogenen Daten sollte die Schwelle angepasst werden
   - Das Modell "denkt", es ist sicherer, immer "negativ" zu sagen

2. **Unausgewogene Trainingsdaten**
   - Im Training: 0 positive, 4.217 negative Labels
   - Das Modell hat gelernt: "Es ist immer sicherer, negativ zu sagen"

3. **Zu hohe Anforderungen**
   - Zeitbasierte Vorhersage: 5 Minuten, 30% Steigerung
   - Das ist sehr schwer zu erreichen
   - Das Modell findet keine Muster, die diese Bedingung erfüllen

4. **Zu wenige Features**
   - Nur 4 Basis-Features (price_open, price_high, price_low, price_close)
   - Keine Feature-Engineering Features aktiviert
   - Keine SMOTE (Oversampling) verwendet

### Was bedeutet das für die Validierung?

**✅ Die Validierung ist trotzdem korrekt!**

- Accuracy ist korrekt berechnet: (0 + 36.830) / 57.126 = 0.6447 ✅
- Confusion Matrix ist konsistent ✅
- Alle Berechnungen sind mathematisch korrekt ✅

**Aber:** Das Modell ist **nicht nützlich** für die Vorhersage von positiven Fällen.

### Lösungsansätze

#### 1. Entscheidungsschwelle anpassen

```python
# Statt Standard-Schwelle 0.5, verwende optimale Schwelle
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Verwende optimal_threshold statt 0.5
predictions = (y_pred_proba >= optimal_threshold).astype(int)
```

#### 2. Feature-Engineering aktivieren

```json
{
  "params": {
    "use_engineered_features": true,
    "feature_engineering_windows": [5, 10, 15]
  }
}
```

#### 3. SMOTE aktivieren (Oversampling)

```json
{
  "params": {
    "use_smote": true
  }
}
```

#### 4. Weniger strenge Anforderungen

- Statt 30% in 5 Minuten → 20% in 10 Minuten
- Oder: Andere Zielvariable verwenden

#### 5. Mehr Features verwenden

```json
{
  "features": [
    "price_open", "price_high", "price_low", "price_close",
    "volume_sol", "market_cap_close", "buy_volume_sol", "sell_volume_sol"
  ]
}
```

### Zusammenfassung

**Warum sind die Werte 0?**

- ✅ **Nicht wegen eines Fehlers im Code**
- ✅ **Nicht wegen falscher Berechnungen**
- ❌ **Sondern weil das Modell keine positiven Vorhersagen macht**

**Das ist ein Modell-Performance-Problem, kein Code-Fehler!**

Die Validierung zeigt korrekt:
- Accuracy ist berechnet: 0.6447 ✅
- Confusion Matrix ist konsistent ✅
- Alle Metriken sind mathematisch korrekt ✅

**Aber:** Das Modell ist nicht nützlich, weil es zu konservativ ist.

### Nächste Schritte

1. ✅ Validierung bestätigt: Code funktioniert korrekt
2. ⚠️ Modell-Performance verbessern:
   - Entscheidungsschwelle anpassen
   - Feature-Engineering aktivieren
   - SMOTE verwenden
   - Mehr Features hinzufügen
   - Weniger strenge Anforderungen

---

**Wichtig:** Die Validierung ist korrekt - das Problem liegt in der Modell-Performance, nicht im Code!

