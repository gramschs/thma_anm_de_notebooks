---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# 8.1 Kostenfunktion und Gradient Descent

In Kapitel 7 haben wir Ableitungen numerisch aus diskreten Messwerten berechnet.
In diesem Kapitel wenden wir genau dieses Werkzeug an, um Modellparameter aus
Messdaten zu bestimmen. *Wie findet man den Parameterwert eines physikalischen
Modells, der am besten zu einer Messreihe passt?*

Als Beispiel betrachten wir die Abkühlkurve eines Aluminiumteils, das frisch aus
der Gusspresse kommt. Das physikalische Modell lautet

$$T(t) = T_\infty + (T_0 - T_\infty) \cdot e^{-\lambda t},$$

wobei $T_\infty$ die Umgebungstemperatur, $T_0$ die Starttemperatur und $\lambda$
die **Abkühlkonstante** ist. $T_\infty$ und $T_0$ sind bekannt. $\lambda$ ist
materialabhängig und muss aus den Sensordaten geschätzt werden. Das Verfahren
dafür heißt **Gradient Descent** und bildet das Fundament moderner
Optimierungsalgorithmen in der Regelungstechnik, der Simulation und dem
maschinellen Lernen.

## Lernziele

* [ ] Sie können erklären, was eine **Kostenfunktion** ist und was ihr Minimum
  bedeutet.
* [ ] Sie können den **mittleren quadratischen Fehler** (MSE) als Kostenfunktion
  implementieren und über dem Parameterraum visualisieren.
* [ ] Sie können den **Gradient-Descent-Algorithmus** in Python implementieren
  und auf ein Einparameter-Problem anwenden.
* [ ] Sie können erklären, welche Rolle die **Lernrate** spielt, und den Effekt
  einer zu großen oder zu kleinen Lernrate beschreiben.
* [ ] Sie wissen, was **Konvergenz** bedeutet, und können eine Konvergenzkurve
  interpretieren.

+++

## Die Kostenfunktion: Güte einer Parameterschätzung messen

Wir erzeugen zunächst simulierte Sensordaten. Das normalverteilte Rauschen mit
Standardabweichung 2 °C bildet die Messungenauigkeit eines echten Temperatursensors
nach.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8')

np.random.seed(42)

# --- Bekannte Größen und wahrer Parameterwert ---
# Variablennamen entsprechen der mathematischen Notation im Text
T_0         = 100.0  # Starttemperatur T_0 in °C
T_inf       =  20.0  # Umgebungstemperatur T_inf (T_∞) in °C
lambda_wahr =   0.05 # wahre Abkühlkonstante in 1/s (im echten Experiment unbekannt)

# --- Messzeitpunkte und simulierte Sensordaten ---
# Sensor misst alle 5 Sekunden über 2 Minuten: 24 Messpunkte
t_mess = np.arange(0, 120, 5)

# Modellwerte laut Physik, dann Messrauschen addieren
T_wahr = T_inf + (T_0 - T_inf) * np.exp(-lambda_wahr * t_mess)
T_mess = T_wahr + np.random.normal(0, 2.0, len(t_mess))

# --- Messdaten visualisieren ---
fig, ax = plt.subplots()
ax.scatter(t_mess, T_mess, label='Messwerte (Sensor)')
ax.set_xlabel('Zeit in s')
ax.set_ylabel('Temperatur in °C')
ax.set_title('Abkühlkurve eines Aluminiumteils')
ax.legend()
plt.show()
```

Wir sehen einen abfallenden Verlauf, der dem Modell entspricht. Jetzt brauchen
wir ein Maß dafür, wie gut ein bestimmter Parameterwert $\lambda$ zu diesen Daten
passt. Als Maß verwenden wir den **mittleren quadratischen Fehler** (englisch:
mean squared error, kurz **MSE**):

$$K(\lambda) = \frac{1}{n} \sum_{i=1}^{n} \bigl(T_{\text{mess},i} - T(t_i,\,\lambda)\bigr)^2$$

Die Funktion $K(\lambda)$ heißt **Kostenfunktion**. Je kleiner $K$, desto besser
passt das Modell. Wir implementieren sie und berechnen $K$ für viele mögliche
$\lambda$-Werte.

```{code-cell} python
# --- Modellfunktion mit expliziten Parametern ---
def modell(t, lam, T_inf, T_0):
    """Abkühlmodell: T(t) = T_inf + (T_0 - T_inf) * exp(-lam * t)."""
    return T_inf + (T_0 - T_inf) * np.exp(-lam * t)

# --- Kostenfunktion: einstellig, weil GD nur lam optimiert ---
# T_inf, T_0, t_mess und T_mess sind für dieses Problem bekannte Konstanten.
# Die Wrapper-Funktion kosten(lam) friert sie ein, damit GD nur lam variiert.
def kosten(lam):
    """MSE zwischen Modell und Messdaten für den Parameter lam.
    Einstellig, damit der GD-Algorithmus sie direkt aufrufen kann."""
    T_vorhergesagt = modell(t_mess, lam, T_inf, T_0)
    # Quadrierte Abweichung für jeden Messpunkt, dann Mittelwert
    return np.mean((T_mess - T_vorhergesagt)**2)

# --- Kostenfunktion für 300 lambda-Werte im Bereich [0.01, 0.12] berechnen ---
lambda_werte = np.linspace(0.01, 0.12, 300)
kosten_werte = []
for lam in lambda_werte:
   kosten_werte.append(kosten(lam))

# --- Kostenfunktion plotten: das Minimum ist direkt sichtbar ---
fig, ax = plt.subplots()
ax.plot(lambda_werte, kosten_werte)
ax.set_xlabel('λ in 1/s')
ax.set_ylabel('K(λ) in °C²')
ax.set_title('Kostenfunktion über dem Parameterraum')
ax.grid(True)
plt.show()
```

Die Kostenfunktion hat ein klares Minimum. Der $\lambda$-Wert an dieser Stelle
ist unsere beste Schätzung. Die naheliegende Strategie wäre, einfach alle
$\lambda$-Werte auf einem feinen Gitter auszuprobieren. Bei einem einzigen
Parameter ist das noch machbar. Bei zwei oder mehr Parametern explodiert die
Rechenzeit: schon bei zehn Parametern mit je 100 Gitterpunkten sind das
$100^{10}$ Auswertungen. Wir brauchen eine gezieltere Suchmethode.

### Mini-Übung 1

1. Was würde $K(\lambda) = 0$ bedeuten? Kann dieser Wert bei echten Messdaten
   mit Sensorrauschen auftreten? Antworten Sie ohne Code.
2. Berechnen Sie $K(0.02)$, $K(0.05)$ und $K(0.10)$. Welcher Wert ist am
   kleinsten? Was schlussfolgern Sie über die Lage des Minimums?
3. Warum nimmt $K(\lambda)$ für $\lambda \to 0$ große Werte an?
   Begründen Sie physikalisch in einem Satz, ohne Code auszuführen.

```{code-cell} python
# Code-Zelle
```

## Gradient Descent: dem Gefälle der Kostenfunktion folgen

Wir suchen das Minimum von $K(\lambda)$, ohne alle Werte auszuprobieren. Die
Idee: Wenn wir am aktuellen Punkt die Steigung von $K$ kennen, wissen wir, in
welche Richtung $K$ abnimmt. Wir machen einen kleinen Schritt in genau diese
Richtung und wiederholen das Vorgehen.

Die Steigung berechnen wir mit der zentralen Differenz aus Kapitel 7. Der
**Gradient-Descent-Algorithmus** lautet damit:

$$\lambda_\text{neu} = \lambda_\text{alt} - \alpha \cdot K'(\lambda_\text{alt})$$

Das Minus sorgt dafür, dass wir entgegen dem Gradienten gehen, also bergab. Der
Parameter $\alpha > 0$ heißt **Lernrate** und bestimmt die Schrittgröße.

```{code-cell} python
# --- Numerische Ableitung der Kostenfunktion ---
# Zentrale Differenz aus Kapitel 7, hier auf die Kostenfunktion angewandt
def ableitung_kosten(lam, dx=1e-6):
    """Ableitung K'(lam), berechnet mit der zentralen Differenz."""
    return (kosten(lam + dx) - kosten(lam - dx)) / (2 * dx)

# --- Gradient-Descent-Algorithmus ---
lam_aktuell   = 0.01   # Startwert: bewusst weit vom Minimum entfernt
alpha         = 1e-8   # Lernrate: muss zur Skala der Kostenfunktion passen
n_iterationen = 1000   # feste Anzahl von Schritten

kosten_verlauf = []    # Kostenwert nach jedem Schritt speichern

for i in range(n_iterationen):
    # Kostenwert vor dem Schritt festhalten
    kosten_verlauf.append(kosten(lam_aktuell))
    # Ableitung am aktuellen Punkt: zeigt die Richtung des stärksten Anstiegs
    grad = ableitung_kosten(lam_aktuell)
    # Schritt entgegen dem Gradienten: bergab auf der Kostenfunktion
    lam_aktuell = lam_aktuell - alpha * grad

print(f"Geschätztes λ:  {lam_aktuell:.5f} 1/s")
print(f"Wahrer Wert λ:  {lambda_wahr:.5f} 1/s")
print(f"MSE am Ende:    {kosten(lam_aktuell):.3f} °C²")
```

Der Algorithmus findet einen Wert nahe am wahren $\lambda$. Die kleine Abweichung
entsteht durch das Sensorrauschen: mit verrauschten Daten ist die beste Schätzung
nicht identisch mit dem wahren Parameter.

### Mini-Übung 2

1. In welche Richtung ändert sich $\lambda$, wenn $K'(\lambda) > 0$?
   Begründen Sie anhand der Update-Formel, ohne Code auszuführen.
2. Setzen Sie `alpha = 1e-4` (1000-mal größer als im Beispiel) und
   starten Sie den Algorithmus neu. Was beobachten Sie?
3. Setzen Sie `alpha = 1e-10` (sehr klein) und starten Sie neu.
   Was beobachten Sie, und warum ist dieses Verhalten ein Problem?

```{code-cell} python
# Code-Zelle
```

## Lernrate und Konvergenz

Wir haben die Kostenwerte in `kosten_verlauf` nach jedem Schritt gespeichert.
Diese **Konvergenzkurve** zeigt, wie sich der Algorithmus dem Minimum nähert.

```{code-cell} python
# --- Konvergenzkurve: Kostenwert über alle Iterationen ---
fig, ax = plt.subplots()
ax.plot(kosten_verlauf)
ax.set_xlabel('Iteration')
ax.set_ylabel('K(λ) in °C²')
ax.set_title('Konvergenz des Gradient-Descent-Algorithmus')
ax.grid(True)
plt.show()

print(f"Kostenwert zu Beginn: {kosten_verlauf[0]:.1f} °C²")
print(f"Kostenwert am Ende:   {kosten_verlauf[-1]:.1f} °C²")
```

Der Kostenwert fällt zunächst schnell und flacht dann ab. Das Abflachen zeigt,
dass der Gradient in der Nähe des Minimums klein wird und die Schritte immer
kürzer werden. Das Plateau ist das Minimum der Kostenfunktion.

Eine sinnvolle Abbruchbedingung lautet: Stopp, wenn sich $\lambda$ kaum noch
ändert, also $|\lambda_\text{neu} - \lambda_\text{alt}| < \varepsilon$ für ein
kleines $\varepsilon$. In der Praxis kombiniert man das mit einer maximalen
Iterationszahl als Sicherheitsgrenze gegen Endlosschleifen.

### Mini-Übung 3

1. Lesen Sie aus dem Konvergenzplot ab: nach ungefähr wie vielen Iterationen
   ist der Kostenwert auf die Hälfte des Startwerts gesunken? Schätzen Sie
   ab, ohne zu rechnen.
2. Starten Sie die GD-Schleife von `lam_aktuell = 0.09` (rechts vom Minimum) und
   zeichnen Sie die Konvergenzkurve. Konvergiert der Algorithmus zum selben
   Ergebnis?
3. Was würde der Algorithmus tun, wenn der Startwert bereits exakt dem Minimum
   entspräche? Begründen Sie ohne Code.

```{code-cell} python
# Code-Zelle
```

## Zusammenfassung und Ausblick

Die Kostenfunktion $K(\lambda)$ misst die Güte einer Parameterschätzung als
mittleren quadratischen Fehler zwischen Modell und Messdaten. Ihr Minimum ist
die beste Schätzung für den gesuchten Parameter. Gradient Descent findet dieses
Minimum durch iterative kleine Schritte entgegen dem Gradienten. Die Ableitung
berechnen wir mit der zentralen Differenz aus Kapitel 7, hier direkt als
Werkzeug eingesetzt. Die Lernrate $\alpha$ bestimmt die Schrittgröße und muss
sorgfältig gewählt werden: zu groß führt zu Divergenz, zu klein zu unnötig
vielen Iterationen.

In Abschnitt 8.3 erweitern wir diesen Ansatz auf zwei unbekannte Parameter.
Sowohl $T_\infty$ als auch $\lambda$ werden dann gleichzeitig aus den Messdaten
geschätzt. Das Prinzip bleibt identisch, der Gradient wird zum Vektor zweier
partieller Ableitungen.
