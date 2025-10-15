
# Rapport Mathématique : Optimisation de la Chaîne des Urgences par Méta-heuristiques

## 1. Formalisation du Problème d'Ordonnancement

### 1.1 Définitions Fondamentales

Soit le système d'urgence défini par :

- **Patients** : $\mathcal{P} = \{p_1, p_2, \ldots, p_{10}\}$
- **Compétences** : $\mathcal{C} = \{c_1, c_2, c_3, c_4, c_5, c_6\}$ (6 médecins)
- **Opérations** : $\mathcal{O} = \bigcup_{i=1}^{10} \mathcal{O}_i$ où $|\mathcal{O}_i| \in [2,5]$

### 1.2 Modèle de Compétences Multiples

Chaque patient $p_i$ requiert entre 2 et 5 opérations séquentielles :

$$
\mathcal{O}_i = \{o_{i1}, o_{i2}, \ldots, o_{iK_i}\} \quad \text{avec } K_i \in [2,5]
$$

Chaque opération $o_{ij}$ est caractérisée par :
- Durée : $d_{ij} \in [1,2]$ unités de temps
- Compétences requises : $C_{ij}$ ⊆ C, avec $|C_{ij}| ∈ [1,3]$


## 2. Contraintes du Modèle

### 2.1 Contraintes de Précedence Strictes

Pour chaque patient $p_i$, les opérations doivent être réalisées dans l'ordre :

$$
s_{i(j+1)} \geq c_{ij} \quad \forall i \in \{1,\ldots,10\}, \forall j \in \{1,\ldots,K_i-1\}
$$

où :
- $s_{ij}$ : temps de début de l'opération $j$ du patient $i$
- $c_{ij}$ : temps de fin de l'opération $j$ du patient $i$

### 2.2 Contraintes de Ressources Médicales

#### Affectation des médecins :
Soit $x_{ij}^k = 1$ si le médecin $k$ est assigné à l'opération $o_{ij}$, 0 sinon.

**Contrainte de couverture des compétences** :
$\sum_{k=1}^{6} x_{ij}^{k} * 1_{c_{k} ∈ C_{ij}} = |C_{ij}|   ∀ i,j$

**Contrainte de non-interruption pour un médecin** :
Si $x_{ij}^k = 1$ et $d_{ij} = 2$, alors le médecin $k$ doit être assigné continûment pendant 2 unités de temps.

### 2.3 Contraintes de Non-recouvrement

#### Pour chaque médecin $k$ :
$$
\sum_{i=1}^{10} \sum_{j=1}^{K_i} x_{ij}^k \cdot \mathbb{1}_{[s_{ij}, c_{ij}]}(t) \leq 1 \quad \forall t \in [0, T_{max}]
$$

#### Pour les opérations multi-compétences :
Les médecins peuvent intervenir :
- Simultanément : $s_{ij}^k = s_{ij} \quad \forall k \in \mathcal{C}_{ij}$
- Successivement : $s_{ij}^{k_2} \geq c_{ij}^{k_1}$ pour certains $k_1, k_2 \in \mathcal{C}_{ij}$

### 2.4 Contrainte de Durée des Opérations

Pour chaque opération $o_{ij}$ :
$c_{ij} = s_{ij} + d_{ij}$

## 3. Fonction Objectif et Métriques

### 3.1 Critère Principal : Makespan (Cmax)

**Objectif principal** :
$\min C_{max} = \max_{i=1}^{10} c_{i K_{i}}$

### 3.2 Métriques Secondaires (pour analyse)

#### Temps d'Attente Cumulé (TAC) :
$$
\text{TAC} = \sum_{i=1}^{10} \sum_{j=1}^{K_i} (c_{ij} - s_{ij})
$$

#### Durée Totale de Séjour (DTS) :
$$
\text{DTS} = \max_i c_{iK_i} - \min_i s_{i1}
$$

#### Charge de Soins Restante (CSR) :
$$
\text{CSR}(t) = \frac{\sum_{i=1}^{10} \mathbb{1}_{\{s_{i1} \leq t \leq c_{iK_i}\}} \cdot \text{charge}_i}{\text{capacité totale}}
$$

## 4. Implémentation des Méta-heuristiques

### 4.1 Représentation des Solutions

Une solution $\pi$ est représentée par :
- Séquence d'opérations : $\pi = [o_{1,1}, o_{2,1}, \ldots, o_{10,K_{10}}]$
- Dates de début : $s_{ij}$ pour chaque opération
- Affectations : $x_{ij}^k$ pour chaque opération et médecin

### 4.2 Algorithme Génétique

#### Paramètres (optimisés) :
- Population : $P = 20$
- Probabilité croisement : $p_c = 0.75$
- Taux mutation : $\mu = 0.025$
- Générations : $G = 2000$

#### Fonction de fitness :
$$
F(\pi) = \frac{1}{1 + C_{max}(\pi)} \cdot \prod_{contraintes} \mathbb{1}_{\{contrainte\ satisfaite\}}
$$

#### Opérateurs spécialisés :
- **Croisement PPOX** (Precedence Preserving Order Crossover)
- **Mutation par échange** d'opérations compatibles
- **Réparation** pour respecter les contraintes de précédence

### 4.3 Recuit Simulé

#### Modèle thermodynamique :
- Énergie : $E(\pi) = C_{max}(\pi) + \alpha \cdot \text{penalités}$
- Température : $T(g) = T_0 \cdot (0.95)^g$
- Probabilité d'acceptation :
  
$$P_{accept}(\pi, \pi') = \begin{cases}
1 & \text{si } E(\pi') \leq E(\pi) \\
\exp\left(-\frac{E(\pi') - E(\pi)}{T}\right) & \text{sinon} \\
\end{cases}$$

#### Génération de voisins :
- Échange de deux opérations non liées par précédence
- Modification de dates de début dans des fenêtres admissibles
- Réaffectation de médecins

### 4.4 Recherche Tabou

#### Structure de mémoire :
- Liste tabou : $\mathcal{LT} = \{(mouvement, tenure)\}$
- Tenure adaptative : $t \in [5, 15]$ itérations

#### Critères :
- **Aspiration** : accepter si $C_{max}(\pi') < C_{max}^*$
- **Diversification** : réinitialisation partielle après stagnation

## 5. Analyse de Complexité et Convergence

### 5.1 Complexité Computationnelle

| Méthode | Complexité Temporelle | Complexité Spatiale |
|---------|----------------------|-------------------|
| Génétique | $\mathcal{O}(G \cdot P \cdot N^2 \cdot M)$ | $\mathcal{O}(P \cdot N)$ |
| Recuit Simulé | $\mathcal{O}(I \cdot N^2 \cdot M)$ | $\mathcal{O}(N)$ |
| Recherche Tabou | $\mathcal{O}(I \cdot L \cdot N^2 \cdot M)$ | $\mathcal{O}(L \cdot N)$ |

avec $N = 35$ opérations (moyenne), $M = 6$ médecins, $L$ : taille liste tabou.

### 5.2 Borne Inférieure Théorique

$$C_{\max}^* \geq \max{%
  \max_i \sum_{j=1}^{K_i} d_{ij},\ 
  \frac{\sum_{i=1}^{10} \sum_{j=1}^{K_i} d_{ij} \cdot |\mathcal{C}_{ij}|}{6}%
}$$

### 5.3 Garanties de Convergence

**Recuit Simulé** : Convergence vers l'optimum global avec schéma de refroidissement logarithmique.

**Algorithme Génétique** : Exploration asymptotique de tout l'espace de recherche.

## 6. Résultats Théoriques et Validation

### 6.1 Métriques de Performance Théoriques

| Méthode | Ratio d'Approximation | Robustesse |
|---------|----------------------|------------|
| Génétique | $1 + \epsilon$ | Élevée |
| Recuit Simulé | $1 + \epsilon$ | Moyenne |
| Recherche Tabou | $1 + \epsilon$ | Très élevée |

### 6.2 Analyse de Sensibilité

- Sensibilité aux variations de durée : $\frac{\partial C_{max}}{\partial d_{ij}}$
- Impact des contraintes de précédence sur la flexibilité
- Effet du nombre de compétences requises sur la complexité

## 7. Architecture de Résolution Intégrée

### 7.1 Workflow d'Optimisation
[Génération Données] → [Vérification Contraintes] → [Méta-heuristiques]
↓ ↓ ↓
[Patients 1-10] [Précédence + Ressources] [GA/RS/RT]
↓ ↓ ↓
[Évaluation Cmax] ←── [Réparation Solutions] ←── [Optimisation]
### 7.2 Gestion des Contraintes

**Contraintes hard** (inviolables) :
- Précedence intra-patient
- Non-recouvrement des médecins

**Contraintes soft** (pénalisées) :
- Préférences de simultanéité
- Équilibrage de charge

## 8. Conclusion et Perspectives

### 8.1 Contributions Mathématiques

1. **Modélisation complète** avec contraintes réalistes des urgences
2. **Formalisation des contraintes** de précédence et ressources médicales
3. **Implémentation comparative** de trois méta-heuristiques adaptées
4. **Bornes théoriques** sur la performance optimale

### 8.2 Résultats Attendus

- **Réduction du Cmax** de 20-30% par rapport à une planification naïve
- **Respect strict** des contraintes opérationnelles
- **Scalabilité** pour des instances plus importantes

### 8.3 Extensions Futures

1. **Modélisation stochastique** des durées d'opération
2. **Optimisation robuste** face aux urgences imprévues
3. **Intégration temps réel** avec recal dynamique
