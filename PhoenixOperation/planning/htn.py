from __future__ import annotations
from planning.utils import Queue

from planning.pddl import Action, Problem, apply_action, get_all_groundings, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------
def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    initial_plan = [h for h in hlas if h.name == "FullRescueMission"][0]
    cola = Queue()
    cola.push([initial_plan])

    while not cola.isEmpty():
        plan = cola.pop()

        # Limitar longitud para evitar explosión
        if len(plan) > 50:
            continue

        if is_plan_primitive(plan):
            state = problem.initial_state
            valido = True
            for action in plan:
                if not is_applicable(state, action):
                    valido = False
                    break
                state = apply_action(state, action)
            if valido and problem.isGoalState(state):
                return plan
            continue

        for i, step in enumerate(plan):
            if not is_primitive(step):
                for refinement in step.refinements:
                    nuevo_plan = plan[:i] + refinement + plan[i+1:]
                    cola.push(nuevo_plan)
                break

    return []



# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------

def find_path(start, end, moves_by_from):
    if start == end:
        return []
    cola = Queue()
    cola.push((start, []))
    visited = {start}
    while not cola.isEmpty():
        current, path = cola.pop()
        for action in moves_by_from.get(current, []):
            next_cell = None
            for f in action.add_list:
                if f[0] == "At" and f[1] == "robot":
                    next_cell = f[2]
                    break
            if next_cell is None or next_cell in visited:
                continue
            new_path = path + [action]
            if next_cell == end:
                return new_path
            visited.add(next_cell)
            cola.push((next_cell, new_path))
    return None
def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    all_actions = get_all_groundings(problem.domain, problem.objects)

    moves, pickup_supply, setup_supply, pickup_patient, putdown_actions, rescue_actions = [], [], [], [], [], []

    for a in all_actions:
        name = a.name.lower()
        if "move" in name:
            moves.append(a)
        elif "pickup" in name:
            for f in a.add_list:
                if f[0] == "Holding":
                    (pickup_supply if "supplies" in str(f[2]) else pickup_patient).append(a)
        elif "setup" in name:
            setup_supply.append(a)
        elif "putdown" in name:
            putdown_actions.append(a)
        elif "rescue" in name:
            rescue_actions.append(a)

    # Índice moves por celda origen
    moves_by_from = {}
    for a in moves:
        for f in a.precond_pos:
            if f[0] == "At" and f[1] == "robot":
                moves_by_from.setdefault(f[2], []).append(a)
                break

    # Posición inicial del robot
    robot_start = None
    for f in problem.initial_state:
        if f[0] == "At" and f[1] == "robot":
            robot_start = f[2]
            break

    # Puestos médicos
    medical_cells = [f[1] for f in problem.initial_state if f[0] == "MedicalPost"]

    patients = problem.objects["patients"]
    supplies = problem.objects["supplies"]

    missions = []
    for i, patient in enumerate(patients):
        supply = supplies[i % len(supplies)]

        pick_s = [a for a in pickup_supply if supply in a.name]
        setup_s = [a for a in setup_supply if supply in a.name]
        pick_p = [a for a in pickup_patient if patient in a.name]

        PS = HLA(f"PrepareSupplies_{patient}")
        PS.refinements = []
        for pick in pick_s:
            supply_cell = None
            for f in pick.precond_pos:
                if f[0] == "At" and f[1] != "robot":
                    supply_cell = f[2]
                    break
            if supply_cell is None:
                continue
            for setup in setup_s:
                medical_cell = None
                for f in setup.precond_pos:
                    if f[0] == "MedicalPost":
                        medical_cell = f[1]
                        break
                if medical_cell is None:
                    continue
                start = robot_start if i == 0 else medical_cells[0]
                path1 = find_path(start, supply_cell, moves_by_from)
                path2 = find_path(supply_cell, medical_cell, moves_by_from)
                if path1 is not None and path2 is not None:
                    PS.refinements.append(path1 + [pick] + path2 + [setup])

        EP = HLA(f"ExtractPatient_{patient}")
        EP.refinements = []
        for pick in pick_p:
            patient_cell = None
            for f in pick.precond_pos:
                if f[0] == "At" and patient in str(f[1]):
                    patient_cell = f[2]
                    break
            if patient_cell is None:
                continue
            for putdown in putdown_actions:
                putdown_cell = None
                for f in putdown.precond_pos:
                    if f[0] == "At" and f[1] == "robot":
                        putdown_cell = f[2]
                        break
                for rescue in rescue_actions:
                    rescue_cell = None
                    for f in rescue.precond_pos:
                        if f[0] == "MedicalPost":
                            rescue_cell = f[1]
                            break
                    if putdown_cell != rescue_cell:
                        continue
                    for med_cell in medical_cells:
                        path1 = find_path(med_cell, patient_cell, moves_by_from)
                        path2 = find_path(patient_cell, rescue_cell, moves_by_from)
                        if path1 is not None and path2 is not None:
                            EP.refinements.append(path1 + [pick] + path2 + [putdown, rescue])

        FM = HLA(f"FullRescueMission_{patient}")
        FM.refinements = [[PS, EP]]
        missions.append(FM)

    FullRescueMission = HLA("FullRescueMission")
    FullRescueMission.refinements = [missions]

    Navigate = HLA("Navigate")
    Navigate.refinements = [[m] for m in moves]

    return [FullRescueMission] + missions + [Navigate]