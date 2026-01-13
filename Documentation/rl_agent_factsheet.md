# RL Agent Factsheet

## Überblick
- **Agentklasse:** `jass.agents.rl_agent.RLAgent`
- **Paradigma:** Actor-Critic (PPO)
- **Beobachtungsumfang:** 117 Features (Hand, gespielte Karten, aktueller Stich, Trumpf-One-Hot, normalisierte Stichanzahl + Teampunkte)
- **Aktionsraum:** 36 diskrete Kartenindizes (aller Karten)

## Architektur

```
                          Input (117d)
                              |
                      [FC: 117 → 128]
                          + ReLU
                              |
                      [FC: 128 → 128]
                          + ReLU
                          /         \
                         /           \
            [Policy-Kopf]             [Value-Kopf]
            [FC: 128 → 36]            [FC: 128 → 1]
                 |                         |
            Logits (36)              Value (scalar)
                 |
         [Masking: -∞ für
          ungültige Züge]
                 |
         Categorical Dist
                 |
            [Sampling]
           Action a_t
```

**Reward-Struktur:** Das Reward-System arbeitet auf zwei Zeitskalen. Auf Trick-Ebene wird nach jedem vollendeten Stich in `finalize_trick()` ein Zwischenreward berechnet (optional basierend auf Stichpunkte-Differenz oder Gewinn des Stichs). Dieser wird in der Trajektorie gespeichert und hilft dem Agenten, taktische Ziele zu lernen. Nach Spielende aggregiert `finalize_episode()` alle Transitionen und addiert den Terminal-Reward (Team-Punkt-Differenz des gesamten Spiels). Dieses Signal triggert dann den PPO-Update, wenn genug Episoden gesammelt wurden.

### Trumpfwahl-Strategie (Rule Based)

Die Trumpf-Entscheidung wird durch eine Regel-basierte Heuristik bestimmt, die die Jass-Spielweise „68er Punkte oder Schieber" implementiert. Diese Strategie evaluiert alle sechs möglichen Trumpf-Optionen anhand der eigenen Handkarten und ermittelt, welche Option die höchste Punkteerwartung liefert. Im Fall von 68 oder mehr garantierten Punkten wird diese Farbe gespielt; sonst wird Schieber angeboten. Wenn bereits geschoben wurde, wird die Farbe mit den höchsten Punkten gewählt.

## Training-Script und PPO-Update

### Trainings-Loop und Update-Timing

Training gegen RuleBased und MCTS:
zwei RL Agenten gegen entweder 2 Rule Based (Highest Card) oder 2 MCTS Observation (8 samples, 250 Simulations)

1. **Training-Loop – Pro Epoch:**
   - `arena.build_default_arena()` erstellt Arena mit Agent + Gegner
   - `trainer.run_batch()` spielt **batch_size Spiele (z.B. 64)** mit rotierenden Dealern
   - **Während jedem Spiel (nach jedem Stich):** `finalize_trick()` wird aufgerufen
     - Berechnet Zwischenreward = Stichpunkte für gewinnendes Team, negative Stichpunkte für verlierendes Team
     - Speichert diese Differenz in der Transition (z.B. +14/-14)
     - Speichert Transition (state, action, log_prob, value, **reward_trick**) in Buffer
   - **Nach jedem Spiel:** `finalize_episode(terminal_reward)` wird aufgerufen
     - Addiert Terminal-Reward = Team-Punkt-Differenz des ganzen Spiels zu **allen Transitionen des Spiels**
     - Wenn Buffer **≥ `update_every_episodes` Spiele** (Standard: 9): **PPO-Update triggert**
2. **Gesamtanzahl Spiele:** 100'000 gegen Rule Based, 100'000 gegen MCTS

### PPO-Update (Loss und Mini-Batching)

Wenn `update_every_episodes` Spiele gesammelt wurden (z.B. 9 Spiele × 9 Tricks/Spiel = **81 Transitionen**):

**Update-Prozess (ppo_epochs=4 Durchläufe):**
```
Total Transitionen: 81
Mini-Batch Size: 64

PPO-Epoche 1:
  ├─ Reshuffle alle 81 Transitionen zufällig
  ├─ Mini-Batch 1 (64 Transitionen): Loss berechnen → Backprop
  └─ Mini-Batch 2 (17 Transitionen): Loss berechnen → Backprop

PPO-Epoche 2:
  ├─ Reshuffle alle 81 Transitionen
  ├─ Mini-Batch 1: Backprop
  └─ Mini-Batch 2: Backprop

... (Epochen 3 & 4 wiederholen)

Total Gradient Updates: 4 Epochen × 2 Mini-Batches = 8 Updates
```
**Wichtig:** Jeder der 2 RL-Agenten (North & South) führt sein eigenes PPO-Update mit seinen eigenen 81 Transitionen durch. Die Updates sind unabhängig. Obwohl beide Agenten im gleichen Spiel spielen, haben ihre States unterschiedliche Observations (jeder sieht nur seine eigenen Karten), aber **identische Rewards** (beide bekommen das gleiche Team-Signal). Im Nachhinein etwas suboptimale mini-batch size.

Danach: **Buffer geleert, neue Spiele sammeln beginnt.** 

**Loss-Funktion pro Mini-Batch:**
```
1. Compute Returns & Advantages via GAE
   - advantages = (reward + γ*V(next_state) - V(current_state))
   - returns = advantages + V(current_state)

2. PPO Loss = Policy_Loss + 0.5*Value_Loss - 0.001*Entropy
   
   wobei:
   ├─ Policy_Loss = -min(ρ*A, clip(ρ, 1±0.2)*A)
   │  ρ = exp(log_π_new - log_π_old)
   │  A = normalized_advantages
   │
   ├─ Value_Loss = MSE(V_pred(s), target_return)
   │
   └─ Entropy = -π*log(π)  (Explorations-Bonus)
```


## MCTS-Agent (Gegner)

Der RLAgent trainiert gegen einen **Monte-Carlo Tree Search (MCTS)** basierten Agenten mit Imperfect-Information Handling (PIMC – Probabilistic Intelligent Monte Carlo). Der Agent nutzt eine determinization-basierte Strategie:

**Mehrere Bäume durch Sampling:** Für jede Entscheidung werden typischerweise **8 verschiedene Sampling** der unvollständigen Spielinformation durchgeführt. Jedes Sampling erzeugt eine vollständige, konsistente Gamestate der versteckten Karten (gegnerische Hände), die mit der bekannten Observation vereinbar ist. Für jede dieser 8 Determinisierungen wird ein separater MCTS-Baum gebaut und durchsucht.

**Simulationen und Baumwachstum pro Sample:** Jeder Baum führt **150 Simulationen** durch (konfigurierbar). Der Baum wächst dabei inkrementell: Bei jeder Simulation wird am vielversprechendsten Knoten (selektiert via UCB) maximal **ein neuer Kind-Knoten** für einen unerforschten Zug erstellt. Von diesem neuen Knoten wird eine Art random Roll-Out gespielt und dann der Wert des simulierten Knotens aktualisiert.

**Aggregation und Voting:** Nach allen Simulationen in jedem Baum wird der beste Zug für diesen Sample ermittelt. Über alle 8 Samples hinweg wird gezählt, wie viele Bäume für denselben Zug votieren. Der Zug mit den **meisten Votes** wird gespielt (bei Gleichstand: zufällig). Dies ist robuster als einen einzelnen Baum zu nutzen, da Ausreißer durch die Mehrheit abgestimmt werden. 

github link: https://github.com/fabian-stettler/JassAgent
