---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# 8.3 Gradient Descent in 2D und Kurvenanpassung

In Abschnitt 8.1 haben wir $\lambda$ aus den Sensordaten bestimmt, während
$T_\infty$ als bekannt vorausgesetzt wurde. In der Praxis ist $T_\infty$ oft
genauso unbekannt wie $\lambda$: Die Umgebungstemperatur in einer
Produktionshalle schwankt, und wir können sie nicht einfach ablesen. *Wie
passt man ein Modell an, wenn mehrere Parameter gleichzeitig unbekannt sind?*

Die Antwort ist dieselbe Idee wie in 1D: wir minimieren die Kostenfunktion.
Der Gradient hat jetzt zwei Komponenten, eine für jeden Parameter.

## Lernziele

* [ ] Sie können die Kostenfunktion $K(T_\infty, \lambda)$ als Konturplot
  darstellen und das Minimum darin lokalisieren.
* [ ] Sie können den **numerischen Gradienten** für zwei Parameter berechnen,
  indem Sie die zentrale Differenz aus Kapitel 7 auf jeden Parameter einzeln
  anwenden.
* [ ] Sie können Gradient Descent für ein Zweiparameter-Problem implementieren,
  ohne Vektornotation zu verwenden.
* [ ] Sie können `scipy.optimize.minimize` auf ein Kurvenanpassungsproblem
  anwenden und das Ergebnis mit einer eigenen GD-Implementierung vergleichen.

+++

## Zwei Parameter, zwei Update-Schritte

Wir starten mit denselben Sensordaten wie in Abschnitt 8.1. Dieses Mal sind
jedoch sowohl $T_\infty$ als auch $\lambda$ unbekannt. $T_0 = 100\,°C$ bleibt
bekannt, weil die Starttemperatur direkt am Sensor ablesbar ist.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8')

np.random.seed(42)

# --- Messdaten aus Abschnitt 8.1 (gleicher Seed ergibt identische Daten) ---
T_0        = 100.0   # Starttemperatur in °C (bekannt)
T_inf_wahr =  20.0   # wahre Umgebungstemperatur in °C (jetzt unbekannt)
lambda_wahr =  0.05  # wahre Abkühlkonstante in 1/s (unbekannt)

t_mess = np.arange(0, 120, 5)
T_wahr = T_inf_wahr + (T_0 - T_inf_wahr) * np.exp(-lambda_wahr * t_mess)
T_mess = T_wahr + np.random.normal(0, 2.0, len(t_mess))

# --- Modell- und Kostenfunktion für zwei unbekannte Parameter ---
def modell(t, lam, T_inf, T_0):
    """Abkühlmodell mit allen Parametern explizit."""
    return T_inf + (T_0 - T_inf) * np.exp(-lam * t)

def kosten(T_inf, lam):
    """MSE als Funktion der beiden unbekannten Parameter T_inf und lam."""
    T_vorhergesagt = modell(t_mess, lam, T_inf, T_0)
    return np.mean((T_mess - T_vorhergesagt)**2)

# --- Kostenfunktion als Konturplot über dem 2D-Parameterraum ---
# Für jede Kombination aus T_inf und lam wird K berechnet
T_inf_werte = np.linspace(10, 35, 120)
lam_werte   = np.linspace(0.01, 0.10, 120)

K_gitter = np.array([[kosten(T, l) for l in lam_werte]
                      for T in T_inf_werte])

fig, ax = plt.subplots()
cp = ax.contourf(lam_werte, T_inf_werte, K_gitter, levels=40, cmap='viridis')
plt.colorbar(cp, label='K(T_∞, λ) in °C²')
ax.set_xlabel('λ in 1/s')
ax.set_ylabel('T_inf in °C')
ax.set_title('Kostenfunktion über dem 2D-Parameterraum')
plt.show()
```

Das Minimum liegt im dunkelsten Bereich des Konturplots. Wir suchen genau
diesen Punkt.

In 1D lautete der GD-Schritt $\lambda_\text{neu} = \lambda_\text{alt} -
\alpha \cdot K'(\lambda_\text{alt})$. Für zwei Parameter brauchen wir zwei
solche Schritte, einen für jeden Parameter:

$$T_{\infty,\text{neu}} = T_{\infty,\text{alt}}
  - \alpha_{T_\infty} \cdot \frac{\partial K}{\partial T_\infty}$$

$$\lambda_\text{neu} = \lambda_\text{alt}
  - \alpha_\lambda \cdot \frac{\partial K}{\partial \lambda}$$

Die zwei Lernraten dürfen unterschiedlich sein, weil $T_\infty$ in °C und
$\lambda$ in 1/s gemessen werden. Der Gradient in $\lambda$-Richtung ist wegen
der Zeitfaktoren im Modell deutlich größer als in $T_\infty$-Richtung, sodass
$\alpha_\lambda$ entsprechend kleiner gewählt werden muss.

### Mini-Übung 1

1. Lesen Sie aus dem Konturplot die ungefähren Koordinaten des Minimums ab.
   Stimmen sie mit den wahren Werten $T_\infty = 20\,°\text{C}$ und
   $\lambda = 0.05\,\text{1/s}$ überein?
2. Am Punkt $(T_\infty = 35\,°\text{C},\; \lambda = 0.05\,\text{1/s})$ sagt
   das Modell für große $t$ eine zu hohe Temperatur voraus. Welches Vorzeichen
   hat $\partial K / \partial T_\infty$ an diesem Punkt, und in welche Richtung
   bewegt sich $T_\infty$ im nächsten GD-Schritt? Antworten Sie ohne Code.

```{code-cell} python
# Code-Zelle
```

## Numerischer Gradient als Werkzeug aus Kapitel 7

Wir brauchen die partiellen Ableitungen $\partial K / \partial T_\infty$ und
$\partial K / \partial \lambda$. Statt sie analytisch herzuleiten, berechnen
wir sie numerisch mit der zentralen Differenz aus Kapitel 7. Das Prinzip ist
dasselbe wie dort: wir variieren einen Parameter um ein kleines $\Delta x$ und
halten den anderen fest.

```{code-cell} python
# --- Numerischer Gradient: zentrale Differenz für jeden Parameter einzeln ---
# Dasselbe Verfahren wie in Kapitel 7, jetzt auf die Kostenfunktion angewandt

dx = 1e-5  # kleine Störung für die zentrale Differenz

def dK_dT_inf(T_inf, lam):
    """Partielle Ableitung von K nach T_inf (lam wird festgehalten)."""
    return (kosten(T_inf + dx, lam) - kosten(T_inf - dx, lam)) / (2 * dx)

def dK_dlam(T_inf, lam):
    """Partielle Ableitung von K nach lam (T_inf wird festgehalten)."""
    return (kosten(T_inf, lam + dx) - kosten(T_inf, lam - dx)) / (2 * dx)

# --- Gradienten an einem Testpunkt auswerten ---
T_inf_test = 35.0
lam_test   = 0.02

grad_T_inf = dK_dT_inf(T_inf_test, lam_test)
grad_lam   = dK_dlam(T_inf_test, lam_test)

print(f"Testpunkt: T_inf = {T_inf_test} °C,  lam = {lam_test} 1/s")
print(f"dK/dT_inf = {grad_T_inf:.4f} °C")
print(f"dK/dlam   = {grad_lam:.2f} °C² · s")
print()
# Verhältnis der Gradientenbeträge erklärt, warum unterschiedliche Lernraten nötig sind
print(f"Betrag dK/dlam ist {abs(grad_lam)/abs(grad_T_inf):.0f}-mal größer als Betrag dK/dT_inf")
print("=> alpha_lam muss entsprechend kleiner als alpha_T_inf gewählt werden")
```

Die Ausgabe zeigt, dass der Gradient in $\lambda$-Richtung viel größer ist.
Mit einer einheitlichen Lernrate würde $\lambda$ massive Sprünge machen, während
$T_\infty$ kaum vorankommt. Die unterschiedlichen Lernraten gleichen das aus.

### Mini-Übung 2

1. Berechnen Sie `dK_dT_inf(20.0, 0.05)` (am wahren Parameterwert). Was
   erwarten Sie für das Ergebnis, und warum? Überprüfen Sie Ihre Vorhersage
   mit Python.
2. Berechnen Sie `dK_dT_inf(20.0, 0.05)` und `dK_dlam(20.0, 0.05)`. Sind
   die Werte exakt null? Begründen Sie das Ergebnis in einem Satz.

```{code-cell} python
# Code-Zelle
```

## Vollständige Kurvenanpassung und scipy

Wir haben alle Zutaten: die Kostenfunktion, die Gradienten und die
Update-Regeln. Jetzt setzen wir alles zu einer vollständigen
Kurvenanpassung zusammen und vergleichen das Ergebnis mit
`scipy.optimize.minimize`.

```{code-cell} python
from scipy.optimize import minimize

# --- Gradient-Descent-Algorithmus für zwei Parameter ---
T_inf_aktuell = 35.0   # Startwert: deutlich über wahrem T_inf = 20 °C
lam_aktuell   = 0.02   # Startwert: deutlich unter wahrem λ = 0.05 1/s
alpha_T_inf   = 5e-3   # Lernrate für T_inf (größer wegen kleinem Gradienten)
alpha_lam     = 2e-7   # Lernrate für lam   (kleiner wegen großem Gradienten)
n_iterationen = 2000

kosten_verlauf = []

for i in range(n_iterationen):
    kosten_verlauf.append(kosten(T_inf_aktuell, lam_aktuell))
    # Partielle Ableitungen am aktuellen Punkt berechnen
    grad_T = dK_dT_inf(T_inf_aktuell, lam_aktuell)
    grad_l = dK_dlam(T_inf_aktuell, lam_aktuell)
    # Jeden Parameter unabhängig aktualisieren
    T_inf_aktuell = T_inf_aktuell - alpha_T_inf * grad_T
    lam_aktuell   = lam_aktuell   - alpha_lam   * grad_l

print("--- Gradient Descent ---")
print(f"T_inf = {T_inf_aktuell:.4f} °C   (wahr: {T_inf_wahr:.4f} °C)")
print(f"lam   = {lam_aktuell:.5f} 1/s  (wahr: {lambda_wahr:.5f} 1/s)")
print(f"MSE   = {kosten(T_inf_aktuell, lam_aktuell):.4f} °C²")
print()

# --- scipy.optimize.minimize als Vergleich ---
# scipy erwartet eine Funktion mit einem einzigen Parametervektor
def kosten_vektor(params):
    """Wrapper für scipy: Parameter als Liste [T_inf, lam]."""
    T_inf, lam = params
    return kosten(T_inf, lam)

# sinnvolle Grenzen für die Parameter festlegen
grenzen = [
    (None, None),   # T_inf: unbegrenzt
    (1e-6, 10),     # lam:   muss positiv sein!
]

# minimize() findet das Minimum automatisch, ohne dass wir alpha wählen müssen
ergebnis = minimize(kosten_vektor, x0=[35.0, 0.02], bounds=grenzen)
T_inf_scipy, lam_scipy = ergebnis.x

print("--- scipy.optimize.minimize ---")
print(f"T_inf = {T_inf_scipy:.4f} °C   (wahr: {T_inf_wahr:.4f} °C)")
print(f"lam   = {lam_scipy:.5f} 1/s  (wahr: {lambda_wahr:.5f} 1/s)")
print(f"MSE   = {kosten(T_inf_scipy, lam_scipy):.4f} °C²")
print(f"Iterationen: {ergebnis.nit}")

# --- Vergleichsplot: Messdaten und beide Anpassungen ---
t_fein = np.linspace(0, 120, 300)

fig, ax = plt.subplots()
ax.scatter(t_mess, T_mess, label='Messwerte', zorder=5)
ax.plot(t_fein, modell(t_fein, lam_aktuell, T_inf_aktuell, T_0),
        label='Gradient Descent')
ax.plot(t_fein, modell(t_fein, lam_scipy, T_inf_scipy, T_0),
        linestyle='--', label='scipy')
ax.set_xlabel('Zeit in s')
ax.set_ylabel('Temperatur in °C')
ax.set_title('Kurvenanpassung: GD vs. scipy')
ax.legend()
plt.show()

# --- Konvergenzkurve des GD ---
fig, ax = plt.subplots()
ax.plot(kosten_verlauf)
ax.set_xlabel('Iteration')
ax.set_ylabel('K(T_∞, λ) in °C²')
ax.set_title('Konvergenz des 2D-Gradient-Descent')
ax.grid(True)
plt.show()
```

Beide Methoden finden nahezu dasselbe Ergebnis. `scipy.optimize.minimize`
passt die Schrittgröße automatisch an, braucht keine manuell gewählten
Lernraten und konvergiert deutlich schneller. Die eigene GD-Implementierung
ist dafür transparent: wir sehen jeden Schritt und können den Algorithmus
nach Bedarf anpassen.

**Wann GD selbst implementieren, wann scipy?**
`scipy.optimize.minimize` ist in der Ingenieurpraxis die erste Wahl: robust,
schnell und ohne Lernraten-Tuning. Die eigene GD-Implementierung ist sinnvoll,
wenn man den Algorithmus verstehen, Zwischenergebnisse speichern oder
den Update-Schritt anpassen will, zum Beispiel in der Simulation von
Lernprozessen.
```

### Mini-Übung 3

1. `scipy` meldet in `ergebnis.nit` die Anzahl der benötigten Iterationen.
   Vergleichen Sie diese Zahl mit den 2000 GD-Iterationen. Was schlussfolgern
   Sie über die Effizienz der beiden Methoden?
2. Ändern Sie den Startwert für scipy auf `x0=[50.0, 0.001]` (weit entfernt
   vom Minimum). Findet `scipy` trotzdem das richtige Ergebnis?
3. Versuchen Sie denselben schlechten Startwert mit Ihrem GD-Algorithmus.
   Was beobachten Sie?

```{code-cell} python
# Code-Zelle
```

## Zusammenfassung und Ausblick

Gradient Descent für zwei Parameter funktioniert nach demselben Prinzip wie
in 1D: für jeden Parameter einen Update-Schritt, berechnet mit der zentralen
Differenz aus Kapitel 7. Die Lernraten müssen an die unterschiedlichen Skalen
der Parameter angepasst werden. `scipy.optimize.minimize` löst dasselbe Problem
robuster und schneller, ohne manuelles Lernraten-Tuning.

In Abschnitt 8.4 wenden wir die vollständige Kurvenanpassungs-Pipeline auf
ein neues ingenieurtechnisches Problem an. Kapitel 9 wechselt dann das Thema:
wir kehren zur Frage zurück, wie man das bestimmte Integral einer Funktion
numerisch berechnet, und lernen mit der Trapez- und Simpson-Regel die
klassischen Quadraturmethoden kennen.
