---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# 8.5 Übungen

## Übung 8.5 (✩)

Gegeben ist der folgende Code, der Gradient Descent auf eine
Abkühlkurve anwendet.

```python
import numpy as np

# Messdaten: sechs Temperaturwerte in °C
t_mess = np.array([ 0,  10,  20,  30,  40,  50])
T_mess = np.array([100, 72,  53,  40,  31,  25])
T_0    = 100.0
T_inf  =  20.0

def modell(t, lam):
    return T_inf + (T_0 - T_inf) * np.exp(-lam * t)

def kosten(lam):
    return np.mean((T_mess - modell(t_mess, lam))**2)

def ableitung_kosten(lam, dx=1e-6):
    return (kosten(lam + dx) - kosten(lam - dx)) / (2 * dx)

lam_aktuell = 0.01
alpha = 0.01

for i in range(8):
    grad = ableitung_kosten(lam_aktuell)
    lam_aktuell = lam_aktuell + alpha * grad   # Zeile A
    print(f"i={i+1}: lam = {lam_aktuell:.5f},  K = {kosten(lam_aktuell):.2f}")
```

1. Enthält der Code einen Fehler? Wenn ja: in welcher Zeile, und was
   genau ist falsch? Antworten Sie ohne Code.
2. Welches Vorzeichen hat `grad` in der ersten Iteration? Begründen Sie
   anhand der Lage von `lam_aktuell = 0.01` relativ zum Minimum.
3. Wird der Kostenwert `K` nach der ersten Iteration größer oder kleiner?
   Begründen Sie mit den Ergebnissen aus Fragen 1 und 2.
4. Korrigieren Sie Zeile A, führen Sie den Code aus und überprüfen Sie
   Ihre Vorhersagen.

```{code-cell} python
# Code-Zelle
```

## Übung 8.6 (✩✩)

In einem Mikrobiologielabor wird das Wachstum einer Bakterienkultur gemessen.
Das Modell lautet

$$N(t) = N_0 \cdot e^{r \cdot t},$$

wobei $N_0 = 100$ die bekannte Startpopulation (Zellen pro µL) und $r$ die
unbekannte Wachstumsrate (in 1/h) ist.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8')

np.random.seed(3)

N_0     = 100.0
r_wahr  = 0.30          # wahre Wachstumsrate in 1/h (unbekannt)
t_mess  = np.array([0, 1, 2, 3, 4, 5])     # Messzeitpunkte in h
N_wahr  = N_0 * np.exp(r_wahr * t_mess)
N_mess  = N_wahr + np.random.normal(0, 10, len(t_mess))
```

1. Plotten Sie die Messdaten `N_mess` gegen `t_mess`. Stellen Sie
   dieselben Daten zusätzlich in einem Semilog-Diagramm dar
   (`ax.semilogy`). Was beobachten Sie im Semilog-Diagramm?
2. Implementieren Sie das Modell $N(t, r)$ und die Kostenfunktion
   `kosten(r)` als MSE. Plotten Sie `kosten(r)` für
   $r \in [0.05,\; 0.60]$, um das Minimum zu lokalisieren.
3. Implementieren Sie den Gradient-Descent-Algorithmus mit der zentralen
   Differenz. Wählen Sie `r_start = 0.05` und `n_iterationen = 2000`.
   Probieren Sie verschiedene Werte für `alpha` aus und wählen Sie einen,
   der zuverlässig konvergiert. Geben Sie das geschätzte $r$ aus.
4. Plotten Sie die angepasste Kurve über die Messdaten.
5. Bestimmen Sie $r$ alternativ mit `scipy.optimize.minimize` und
   vergleichen Sie das Ergebnis.

```{code-cell} python
# Code-Zelle
```

## Übung 8.7 (✩✩✩)

Ein Schwingungssensor misst die freie gedämpfte Schwingung eines Maschinenteils
nach einer impulsförmigen Anregung. Das Modell lautet

$$x(t) = A \cdot e^{-\delta t} \cdot \cos(\omega t),$$

wobei $A$ die Anfangsamplitude (in mm), $\delta$ die Abklingrate (in 1/s)
und $\omega$ die Kreisfrequenz (in rad/s) sind.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8')
from scipy.optimize import minimize

np.random.seed(11)

A_wahr     =  10.0   # mm
delta_wahr =   0.5   # 1/s
omega_wahr =   5.0   # rad/s

t_mess = np.linspace(0, 4, 50)
x_wahr = A_wahr * np.exp(-delta_wahr * t_mess) * np.cos(omega_wahr * t_mess)
x_mess = x_wahr + np.random.normal(0, 0.5, len(t_mess))
```

1. Plotten Sie `x_mess` gegen `t_mess`. Schätzen Sie aus dem Plot grob
   die Amplitude $A$, die Abklingrate $\delta$ und die Kreisfrequenz
   $\omega$ ab. (Hinweis: Die Periodendauer $T$ ist im Plot ablesbar,
   und es gilt $\omega = 2\pi / T$.)
2. Implementieren Sie das Modell und die Kostenfunktion `kosten(A, delta,
   omega)`. Implementieren Sie Gradient Descent mit drei unabhängigen
   Update-Schritten und numerischen partiellen Ableitungen. Verwenden
   Sie die Startwerte `A=8, delta=0.3, omega=4` (nahe am wahren Wert)
   und `n_iterationen=5000`. Geben Sie die gefundenen Parameter aus.
3. Setzen Sie den Startwert für $\omega$ auf `omega=8` (weit vom wahren
   Wert). Was beobachten Sie? Konvergiert der Algorithmus zum selben
   Ergebnis?
4. Lösen Sie das Problem mit `scipy.optimize.minimize` von beiden
   Startwerten aus. Vergleichen Sie die Robustheit von GD und scipy.
5. Begründen Sie: Warum hat die Kostenfunktion $K(A, \delta, \omega)$
   mehrere lokale Minima, während die Kostenfunktionen der Abkühlkurve
   (Abschnitt 8.1) und der Wöhlerkurve (Übung 8.4) nur ein einziges
   Minimum besitzen?

```{code-cell} python
# Code-Zelle
```
