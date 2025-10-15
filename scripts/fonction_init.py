from copy import deepcopy
import random
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import math

# --- Global Constants (Defined only once, but passed to functions) ---
random.seed(42)

SKILLS = [1, 2, 3, 4, 5, 6]
NUM_PATIENTS = 10
MAX_OPS = 5  # au plus 5 opérations

DATA = {
    1: {
        1: [(1,2)],
        2: [(1,1),(2,1)],
        3: [(1,1),(3,1)],
        4: [(1,1),(2,2)],  # C1 et C2x2
        5: [(4,1),(5,2),(6,1)],
    },
    2: {
        1: [(2,1),(3,1)],
        2: [(2,1),(3,1)],
        3: [(2,1)],
        4: [],
        5: [],
    },
    3: {
        1: [(3,2)],
        2: [(3,1)],
        3: [],
        4: [],
        5: [],
    },
    4: {
        1: [(4,2)],
        2: [(5,1),(6,1)],
        3: [(6,2)],
        4: [(4,2)],
        5: [(1,1),(2,1)],
    },
    5: {
        1: [(2,2)],
        2: [(5,1)],
        3: [(5,1),(6,1)],
        4: [(4,1),(5,1)],
        5: [(3,1)],
    },
    6: {
        1: [(1,1)],
        2: [(4,1)],
        3: [(6,1)],
        4: [],
        5: [],
    },
    7: {
        1: [(6,2)],
        2: [(1,1)],
        3: [(5,1),(6,1)],
        4: [(3,1)],
        5: [],
    },
    8: {
        1: [(3,1),(5,1)],
        2: [(2,1),(5,1)],
        3: [(3,1),(6,1)],
        4: [(6,1)],
        5: [],
    },
    9: {
        1: [(5,1)],
        2: [(4,1)],
        3: [(1,1)],
        4: [],
        5: [],
    },
    10: {
        1: [(4,1)],
        2: [(4,1),(5,1)],
        3: [(1,1),(2,1)],
        4: [(4,1)],
        5: [],
    },
}

# ---------------------------
# 1) Data Structure and Task Creation
# ---------------------------
def create_task(num_patients, data, max_ops):
  """Crée toutes les tâches et les structures d'indexation."""
  Task = namedtuple("Task", ["i", "j", "s", "p"])
  ALL_TASKS = []  # liste de toutes les tâches
  TASKS_BY_SKILL_STAGE = defaultdict(list)  # (s, j) -> liste des tâches
  PATIENT_LAST_STAGE = {i: 0 for i in range(1, num_patients + 1)}

  for i in range(1, num_patients + 1):
      # Ensure patient i exists in DATA
      if i in data:
          for j in range(1, max_ops + 1):
              # Use .get() for safe access to stages
              ops = data[i].get(j, [])
              if ops:
                  PATIENT_LAST_STAGE[i] = j
              for (s, p) in ops:
                  t = Task(i=i, j=j, s=s, p=p)
                  ALL_TASKS.append(t)
                  TASKS_BY_SKILL_STAGE[(s, j)].append(t)
      else:
          # Handle case where patient ID in range doesn't exist in DATA (if DATA was dynamic)
          pass 

  return ALL_TASKS, TASKS_BY_SKILL_STAGE, PATIENT_LAST_STAGE

# Execute task creation (Global variables are now populated)
ALL_TASKS, TASKS_BY_SKILL_STAGE, PATIENT_LAST_STAGE = create_task(NUM_PATIENTS, DATA, MAX_OPS)

# ---------------------------
# 2) Initial Sequence Building
# ---------------------------
def build_initial_sequences(skills, max_ops, tasks_by_skill_stage):
    """
    Construit une séquence initiale de tâches (ordre naïf : patients croissants).
    """
    INITIAL_ORDER_OVERRIDE = {
    # Exemple : (2,1): [5,2,8,1,7,...]
    # Laisse vide si tu veux l'ordre naïf (1..10).
    }
    seq = {}
    for s in skills:
        for j in range(1, max_ops + 1):
            tasks = tasks_by_skill_stage.get((s, j), [])
            if not tasks:
                continue
            
            # Ordre naïf = patients croissants
            default_order = sorted(tasks, key=lambda t: (t.i))
            
            # Surcharge éventuelle
            if (s, j) in INITIAL_ORDER_OVERRIDE:
                allowed = set(t.i for t in tasks)
                order_patients = [i for i in INITIAL_ORDER_OVERRIDE[(s, j)] if i in allowed]
                # compléter avec ceux non listés, dans l'ordre naïf
                remaining = [t.i for t in default_order if t.i not in order_patients]
                final_order_patients = order_patients + remaining
                
                # reconstruire la liste de tâches dans cet ordre
                patient_to_tasks = defaultdict(list)
                for t in tasks:
                    patient_to_tasks[t.i].append(t)
                # (il n'y a qu'une tâche par (i,s,j) ici, donc c'est direct)
                ordered_tasks = [patient_to_tasks[i][0] for i in final_order_patients]
            else:
                ordered_tasks = default_order
                
            seq[(s, j)] = ordered_tasks
    return seq

# Execute initial sequence building
init_seq = build_initial_sequences(SKILLS, MAX_OPS, TASKS_BY_SKILL_STAGE)

# --------------------------
# 3) Evaluation (construction du planning "par étapes")
# --------------------------
def evaluate_schedule(sequences, skills, num_patients, data, patient_last_stage, max_ops, return_schedule=False):
    """
    sequences: dict (s,j) -> liste ordonnée de Task
    Calcule un planning non préemptif en respectant l'ordre des opérations (par étapes j=1..max_ops).
    Retourne (makespan, details, op_completion) si return_schedule=True, sinon makespan.
    """
    
    # disponibilité des ressources (temps où la ressource s sera libre)
    res_free = {s: 0 for s in skills}
    # fin de l'étape j pour chaque patient, initialisé à 0 pour l'étape 0
    op_completion = { (i, 0): 0 for i in range(1, num_patients + 1) }
    # pour stocker les dates de chaque tâche
    task_times = {}

    for j in range(1, max_ops + 1):
        # on accumule la fin max de l'étape j par patient
        stage_finish = defaultdict(int)
        
        # pour chaque ressource/skill, on parcourt les tâches de l'étape j dans l'ordre imposé
        for s in skills:
            tasks = sequences.get((s, j), [])
            for t in tasks:
                # readiness = fin de l'étape précédente du patient (i, j-1)
                # t.i is patient ID
                ready = op_completion[(t.i, j - 1)]
                
                # début de la tâche = max(ressource libre, patient prêt)
                start = max(res_free[s], ready)
                finish = start + t.p
                
                # mise à jour de la disponibilité de la ressource
                res_free[s] = finish
                
                # mise à jour de la fin max pour le patient à cette étape j
                stage_finish[t.i] = max(stage_finish[t.i], finish)
                
                # stockage des temps de la tâche (i, j, s)
                task_times[(t.i, j, s)] = (start, finish, t.p)
                
        # on fige la fin d'étape j pour tous les patients
        for i in range(1, num_patients + 1):
            if data[i].get(j, []):  # patient i a bien une opération j
                op_completion[(i, j)] = stage_finish[i]
            else:
                # pas d'opération : fin d'étape = fin d'étape précédente (carry)
                op_completion[(i, j)] = op_completion[(i, j - 1)]


    # makespan = max fin de dernière étape existante par patient
    makespan = 0
    for i in range(1, num_patients + 1):
        last_j = patient_last_stage[i]
        makespan = max(makespan, op_completion[(i, last_j)])

    if return_schedule:
        return makespan, task_times, op_completion
    
    return makespan

# --------------------------
# 4) Visualisation du Gantt
# --------------------------
def _patient_colors(num_patients):
    """Palette stable pour n patients (tab20)."""
    # Use colormaps dictionary access
    cmap = plt.colormaps.get_cmap("tab20")
    # Limit number of colors to 20 for 'tab20' if num_patients > 20
    n = min(20, num_patients) 
    return {i+1: cmap(i / 19) for i in range(num_patients)}


def build_gantt_data(task_times, skills):
    """
    Re-formate task_times -> dict skill -> liste d'items triés
    item = dict(start, end, dur, patient, op)
    """
    by_skill = {s: [] for s in skills}
    horizon = 0
    # task_times is indexed by (i, j, s)
    for (i, j, s), (start, finish, p) in task_times.items():
        by_skill[s].append({
            "start": start, "end": finish, "dur": p,
            "patient": i, "op": j
        })
        horizon = max(horizon, finish)
        
    # trier par début
    for s in skills:
        by_skill[s].sort(key=lambda x: (x["start"], x["patient"], x["op"]))
        
    return by_skill, horizon

def plot_gantt(task_times, skills, num_patients, title="Gantt – Planning par compétence",
               figsize=None, annotate=True, save_path=None, dpi=150):
    """
    task_times : dict (i,j,s) -> (start, end, dur) renvoyé par evaluate_schedule(..., return_schedule=True)
    - Une piste (ligne) par compétence s ∈ {1..6}
    - Couleurs par patient
    """
    by_skill, horizon = build_gantt_data(task_times, skills)
    colors = _patient_colors(num_patients)

    if figsize is None:
        # largeur ~ horizon, hauteur ~ nb skills
        figsize = (max(10, horizon * 0.6 / 3), 1.2 * len(skills) + 2) # Adjusted width for better display

    fig, ax = plt.subplots(figsize=figsize)

    lane_height = 0.8
    y_gap = 0.6
    # pour placer la lane s à y = idx (plus la compétence est petite, plus elle est haute)
    y_positions = {s: (len(skills)-idx-1)*(lane_height + y_gap) for idx, s in enumerate(skills)}
    ymin = -0.5
    ymax = max(y_positions.values()) + lane_height + 0.5

    # Dessin des rectangles
    for s in skills:
        y = y_positions[s]
        # bande de fond par piste
        ax.add_patch(Rectangle((0, y - 0.1), horizon, lane_height + 0.2,
                               facecolor=(0,0,0,0.03), edgecolor="none"))
        for it in by_skill[s]:
            start = it["start"]
            dur   = it["dur"]
            # end   = it["end"]
            i     = it["patient"]
            j     = it["op"]
            rect = Rectangle((start, y), dur, lane_height,
                             facecolor=colors[i], edgecolor="black", linewidth=0.7)
            ax.add_patch(rect)
            if annotate:
                label = f"P{i}-O{j}"
                ax.text(start + dur/2, y + lane_height/2, label,
                        ha="center", va="center", fontsize=9)

    # Axes / ticks / labels
    ax.set_xlim(0, math.ceil(horizon) if horizon > 0 else 1) # Ensure xlim is positive
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Temps (unités)")
    ax.set_yticks([y_positions[s] + lane_height/2 for s in skills])
    ax.set_yticklabels([f"Skill {s}" for s in skills])
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Légende des patients
    legend_handles = [Patch(facecolor=colors[i], edgecolor="black", label=f"Patient {i}") for i in range(1, num_patients + 1)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Gantt sauvegardé : {save_path}")
    plt.show()

# --------------------------
# 5) Execution of the corrected logic
# --------------------------

# We use the calculated initial sequences 'init_seq'
# and pass all necessary global variables as parameters.
makespan, task_times, op_completion = evaluate_schedule(
    sequences=init_seq, 
    skills=SKILLS, 
    num_patients=NUM_PATIENTS, 
    data=DATA, 
    patient_last_stage=PATIENT_LAST_STAGE, 
    max_ops=MAX_OPS, 
    return_schedule=True
)

print(f"Makespan = {makespan}")

# Now task_times will be populated and the Gantt chart can be plotted.
if task_times:
    plot_gantt(
        task_times=task_times, 
        skills=SKILLS, 
        num_patients=NUM_PATIENTS,
        title=f"Gantt – Planning initial (Cmax={makespan})",
        figsize=None, 
        annotate=True, 
        save_path=None
    )
else:
    print("No tasks were scheduled. Check DATA configuration.")
