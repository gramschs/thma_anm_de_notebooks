---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# 8.2 Übungen zu Gradient Descent in 1D

## Übung 8.1 (✩)

Gegeben ist der folgende Code, der Gradient Descent auf eine einfache
Kostenfunktion anwendet.

```python
import numpy as np

def kosten(x):
    return (x - 3)**2 + 1

def ableitung_kosten(x, dx=1e-5):
    return (kosten(x + dx) - kosten(x - dx)) / (2 * dx)

x_aktuell = 0.0
alpha = 0.1

for i in range(5):
    grad = ableitung_kosten(x_aktuell)
    x_aktuell = x_aktuell - alpha * grad
    print(f"Iteration {i+1}: x = {x_aktuell:.4f},  K = {kosten(x_aktuell):.4f}")
```

1. Wo liegt das Minimum von $K(x)$? Begründen Sie ohne Code.
2. Berechnen Sie von Hand, was die erste Ausgabezeile (Iteration 1) zeigt.
   Geben Sie `x_aktuell` und `K(x_aktuell)` nach dem ersten Schritt an.
3. Bewegt sich `x_aktuell` in jedem Schritt nach links oder nach rechts?
   Begründen Sie anhand der Update-Formel ohne Code.
4. Führen Sie den Code aus und überprüfen Sie Ihre Vorhersagen aus den
   Fragen 1 bis 3.

```{code-cell} python
# Code-Zelle
```

## Übung 8.2 (✩✩)

Die Funktion $f(x) = x^4 - 4x^2$ hat mehr als ein lokales Minimum.

1. Plotten Sie $f(x)$ im Bereich $x \in [-2.5,\; 2.5]$. Wo liegen die
   lokalen Minima ungefähr? Bestimmen Sie die genauen Positionen analytisch
   mit $f'(x) = 0$.
2. Implementieren Sie Gradient Descent mit der numerischen Ableitung
   (zentrale Differenz). Verwenden Sie `alpha = 0.02` und
   `n_iterationen = 500`. Starten Sie von `x_start = 2.0` und geben Sie
   das gefundene Minimum aus.
3. Wiederholen Sie den Lauf mit `x_start = -2.0`. Was beobachten Sie?
4. Stellen Sie die Konvergenzkurven beider Läufe in einem gemeinsamen Plot
   dar. Beschriften Sie Achsen und Kurven.
5. Starten Sie einen dritten Lauf von `x_start = 0.01`. Zu welchem Minimum
   konvergiert der Algorithmus? Begründen Sie, warum er nicht bei $x = 0$
   stehen bleibt, obwohl $f'(0) = 0$ gilt.

```{code-cell} python
# Code-Zelle
```

