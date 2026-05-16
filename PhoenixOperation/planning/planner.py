from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    
    cola = Queue()
    visitados = set()
    visitados.add(problem.initial_state)
    
    # Iniciamos añadiendo a la cola el estado inicial y un plan vacío
    cola.push((problem.initial_state,[]))
    
    while not cola.isEmpty():
        estado, plan_act = cola.pop()
        
        # Si se llega al objetivo se devuelve el plan
        if problem.isGoalState(estado):
            return plan_act
        
        for sig_estado, accion, costo in problem.getSuccessors(estado):
            # Si el sucesro no fue visitado lo añadimos a la cola
            if sig_estado not in visitados:
                # Ignoramos el costo porque es BFS
                cola.push((sig_estado,plan_act+[accion]))
                # Y marcamos como visitado al siguiente estado
                visitados.add(sig_estado)
            
    # Se exploró todo el espacio alcanzable y no se encontró una solución
    return []
    
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    goal_set = set(goal)
    add_set = set(action.add_list)
    del_set = set(action.del_list)
    pre_pos = set(action.precond_pos)
    
    # 1. Relevancia: La acción debe aportar al menos un fluente necesario para el objetivo actual
    if not (goal_set & add_set):
        return None
        
    # 2. Consistencia: La acción no debe borrar algo que el objetivo necesita
    # (A menos que la misma acción lo borre y lo vuelva a añadir, caso borde manejado por la resta)
    if (goal_set - add_set) & del_set:
        return None
        
    # 3. Regresión: Nuevo Objetivo = (Objetivo Actual - Efectos Positivos) U Precondiciones
    new_goal = (goal_set - add_set) | pre_pos
    return frozenset(new_goal)

    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    start_goal = problem.goal
    
    # 1. Obtener todas las acciones instanciadas (grounded) del problema
    all_actions = get_all_groundings(problem.domain, problem.objects)
    
    # 2. PRE-COMPUTACIÓN: Índice inverso y detección de predicados estáticos
    actions_by_add = {}
    dynamic_predicates = set()
    
    for action in all_actions:
        # Indexar acciones por los fluentes que agregan
        for f in action.add_list:
            actions_by_add.setdefault(f, []).append(action)
            dynamic_predicates.add(f[0])
        # Identificar qué predicados cambian durante el juego
        for f in action.del_list:
            dynamic_predicates.add(f[0])
            
    # Función auxiliar para podar estados lógicamente imposibles (Mutex)
    def is_goal_consistent(g_set: frozenset) -> bool:
        at_counts = {}
        for f in g_set:
            if f[0] == "At": # f = ("At", entidad, ubicacion)
                entidad = f[1]
                at_counts[entidad] = at_counts.get(entidad, 0) + 1
                # Si una entidad debe estar en dos lugares distintos simultáneamente, es imposible
                if at_counts[entidad] > 1:
                    return False
        return True

    # 3. Inicializar Frontera (Cola para BFS) y Lista de Visitados (para Subsunción)
    frontier = Queue()
    frontier.push((start_goal, []))  # Guarda tuplas de (estado_objetivo, plan_construido)
    visited_goals = [start_goal]     # Usamos lista en vez de set para poder iterar y comparar subconjuntos
    
    while not frontier.isEmpty():
        current_goal, plan = frontier.pop()
        
        # CONDICIÓN DE PARADA: Si el objetivo parcial ya es verdad en el mapa original
        if current_goal.issubset(problem.initial_state):
            return list(reversed(plan)) # Invertimos porque construimos el plan hacia atrás
            
        # Buscar acciones relevantes (solo para los fluentes que aún no están resueltos por el estado inicial)
        unsatisfied_fluents = current_goal - problem.initial_state
        candidate_actions = set()
        for fluent in unsatisfied_fluents:
            if fluent in actions_by_add:
                candidate_actions.update(actions_by_add[fluent])
                
        # Expandir la frontera
        for action in candidate_actions:
            raw_new_goal = regress(current_goal, action)
            
            if raw_new_goal is None:
                continue
                
            # --- OPTIMIZACIÓN 1: Limpieza de Fluentes Estáticos (Garbage Collection) ---
            cleaned_goal = set()
            is_impossible = False
            for f in raw_new_goal:
                if f[0] not in dynamic_predicates:
                    # Es un fluente estático (ej. Adjacent, MedicalPost)
                    if f in problem.initial_state:
                        continue # Ya sabemos que es verdad, no lo arrastramos para ahorrar memoria
                    else:
                        # Pide un fluente estático que NO existe en el mapa (ej. pared donde no la hay)
                        is_impossible = True
                        break
                else:
                    cleaned_goal.add(f)
                    
            if is_impossible:
                continue
                
            new_goal = frozenset(cleaned_goal)
            
            # --- OPTIMIZACIÓN 2: Poda de Inconsistencias (Mutex) ---
            if not is_goal_consistent(new_goal):
                continue
                
            # --- OPTIMIZACIÓN 3: Subsunción de Estados (State Subsumption) ---
            # Si ya visitamos un estado que exige menos cosas (es subconjunto), este camino es redundante
            is_subsumed = False
            for v in visited_goals:
                if v.issubset(new_goal):
                    is_subsumed = True
                    break
                    
            if not is_subsumed:
                visited_goals.append(new_goal)
                frontier.push((new_goal, plan + [action]))
                
    # Si la cola se vacía y no encontramos solución
    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###

    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
