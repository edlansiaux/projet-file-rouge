# Rapport : Optimisation de la Chaine des Urgences par Méta-heuristiques

## Contexte et Problématique

Ce rapport présente une modélisation mathématique et computationnelle pour l'optimisation de la chaîne des urgences hospitalières. La problématique centrale concerne la gestion des **tensions** dans les services d'urgence, définies comme un déséquilibre entre le flux de patients et la capacité de prise en charge.

## Modélisation Mathématique du Problème

### Formalisation du Problème d'Ordonnancement

Soit les ensembles suivants :

- **P** = {p₁, p₂, ..., pₙ} : ensemble des patients
- **R** = {r₁, r₂, ..., rₘ} : ensemble des ressources
- **T** = {t₁, t₂, ..., tₖ} : ensemble des tâches de soins
- **C** = {c₁, c₂, ..., cₗ} : ensemble des compétences requises

Chaque patient *pᵢ* a un parcours de soins défini par une séquence ordonnée de tâches :  

\[
Wᵢ = \langle tᵢ₁, \; tᵢ₂, \; \ldots, \; tᵢₛ \rangle
\]

Chaque tâche *tⱼ* nécessite un ensemble de compétences :  

\[
Comp(tⱼ) \subseteq C
\]


### Variables de Décision

- **xᵢⱼᵏ** = 1 si la tâche j du patient i est assignée à la ressource k, 0 sinon
- **sᵢⱼ** : temps de début de la tâche j du patient i
- **cᵢⱼ** : temps de fin de la tâche j du patient i

### Contraintes

1. **Contrainte de précédence** : sᵢⱼ ≥ cᵢ₍ⱼ₋₁₎ ∀i ∈ P, ∀j ∈ {2,...,|Wᵢ|}
2. **Contrainte de ressource**: ∑ xᵢⱼᵏ ≤ 1 ∀k ∈ R, ∀t ∈ [0, T_max]
3. **Contrainte de compétence**:xᵢⱼᵏ = 0 si Comp(rₖ) ∩ Comp(tⱼ) = ∅

### Fonction Objectif

Le **makespan** (durée totale) est défini comme :  

C_max = max{Cᵢ} pour i = 1..n

Où \( Cᵢ \) est le temps d'achèvement du patient \( pᵢ \).  

La fonction objectif à minimiser est :  

min f(π) = C_max + α × ∑ᵢ Wᵢ + β × ∑ⱼ Uⱼ

Où :
- **Wᵢ** = Cᵢ - Aᵢ : temps d'attente du patient i (Aᵢ = temps d'arrivée)
- **Uⱼ** = (temps d'occupation de rⱼ) / T_max : taux d'utilisation de la ressource j
- **α, β** : coefficients de pondération

## Implémentation des Méta-heuristiques

### 1. Algorithme Génétique

#### Représentation Chromosomique

Chromosome = [p₁, p₂, ..., pₙ] où pᵢ ∈ P

#### Opérateurs Génétiques

**Sélection par tournoi** :
P(tournoi) = {random.sample(population, k)}
parent = argmin{f(p) | p ∈ P(tournoi)}

**Croisement OX (Order Crossover)** :
Enfant[a:b] = Parent1[a:b]
Remplissage avec gènes de Parent2 dans l'ordre

**Mutation par swap** :
Pour chaque gène i avec probabilité μ :
j = random.randint(0, n-1)
échanger gène[i] et gène[j]


### 2. Recuit Simulé

#### Fonction d'Acceptation
P(accept) = exp(-ΔE / T) si ΔE > 0
          = 1 si ΔE ≤ 0


Où :
- **ΔE** = f(s') - f(s) : différence de coût
- **T** : température courante

#### Schéma de Refroidissement
T_{k+1} = α × T_k, avec α ∈ [0.9, 0.99]


#### Algorithme
s = s₀
T = T₀
tant que T > T_min:
s' = voisin(s)
ΔE = f(s') - f(s)
si ΔE < 0 ou random() < exp(-ΔE/T):
s = s'
T = α × T


### 3. Recherche Tabou

#### Structure de Mémoire
Liste Tabou LT = {(move, tenure)} avec tenure décroissant

#### Fonction d'Aspiration
Accepté si f(s') < f(s*) même si move ∈ LT

#### Critère d'Arrêt
Max itérations ou stagnation pendant k itérations
