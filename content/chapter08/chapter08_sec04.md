---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# 8.4 Übungen zu Gradient Descent in 2D

## Übung 8.3 (✩)

Gegeben ist der folgende Code, der `scipy.optimize.minimize` auf eine
zweidimensionale Kostenfunktion anwendet.

```python
from scipy.optimize import minimize

def f(params):
    a, b = params
    return (a - 5)**2 + 2 * (b + 3)**2

ergebnis = minimize(f, x0=[0.0, 0.0])

print(f"a          = {ergebnis.x[0]:.4f}")
print(f"b          = {ergebnis.x[1]:.4f}")
print(f"f(a, b)    = {ergebnis.fun:.6f}")
print(f"Konvergiert: {ergebnis.success}")
print(f"Iterationen: {ergebnis.nit}")
```

1. Wo liegt das Minimum von $f(a, b)$? Berechnen Sie es ohne Code und
   geben Sie $a^*$ und $b^*$ an.
2. Was erwarten Sie für die Ausgabe von `ergebnis.x` und `ergebnis.fun`?
   Schreiben Sie die erwarteten Werte auf, bevor Sie den Code ausführen.
3. Der erste Term $(a - 5)^2$ hat den Koeffizienten 1, der zweite
   $2 \cdot (b + 3)^2$ den Koeffizienten 2. In welcher Richtung ist
   $f$ steiler, in $a$- oder in $b$-Richtung? Was würde das für die
   Wahl von Lernraten bei einem eigenen GD-Algorithmus bedeuten?
4. Führen Sie den Code aus und überprüfen Sie Ihre Vorhersagen aus
   den Fragen 1 und 2.

```{code-cell} python
# Code-Zelle
```

## Übung 8.4 (✩✩)

Die **Wöhlerkurve** beschreibt, wie viele Lastspiele $N$ ein Bauteil bei
einer Spannungsamplitude $S$ aushält. Das Modell lautet

$$S = a \cdot N^{-b},$$

wobei $a$ und $b$ materialabhängige Parameter sind. Maschinenbauteile aus
Stahl haben typischerweise $a \approx 1500\,\text{MPa}$ und
$b \approx 0.12$.

Im folgenden Datensatz stehen simulierte Prüfergebnisse zur Verfügung:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8')
from scipy.optimize import minimize

np.random.seed(7)

# --- Wöhlerversuch: Spannungsamplitude vs. Lastspielzahl ---
a_wahr = 1500.0  # MPa
b_wahr = 0.12
N_mess = np.array([1e4, 3e4, 1e5, 5e5, 2e6, 1e7])  # Lastspielzahl
S_wahr = a_wahr * N_mess**(-b_wahr)                  # Spannungsamplitude in MPa
# Multiplikatives Rauschen: ±5 % Streuung
S_mess = S_wahr * (1 + np.random.normal(0, 0.05, len(N_mess)))
```

1. Plotten Sie `N_mess` gegen `S_mess` in einem Log-Log-Diagramm
   (`ax.loglog`). Beschriften Sie die Achsen mit Einheiten. Warum
   erscheint die Wöhlerkurve im Log-Log-Diagramm als Gerade?
2. Implementieren Sie Kostenfunktion und numerische Gradienten für die
   Parameter $a$ und $b$. Wählen Sie `alpha_a = 0.5` und
   `alpha_b = 1e-8` als Lernraten sowie `[500.0, 0.05]` als Startwerte.
   Führen Sie 10000 Iterationen durch und geben Sie die gefundenen
   Parameter aus.
3. Plotten Sie die angepasste Kurve über die Messdaten (Log-Log).
4. Wiederholen Sie die Anpassung mit `scipy.optimize.minimize` und
   vergleichen Sie die Ergebnisse.
5. Warum ist $\alpha_b$ so viel kleiner als $\alpha_a$? Berechnen Sie
   die numerischen Gradienten am Startpunkt und begründen Sie das
   Verhältnis der Lernraten damit.

```{code-cell} python
# Code-Zelle
```
