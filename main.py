import concurrent.futures
import copy
import enum
import math, random
import string
import time

import yaml

from TuringMachine import *

class MutationType(enum.Enum):
    ADD_STATE = enum.auto()
    DUPLICATE_STATE = enum.auto()
    REMOVE_STATE = enum.auto()
    ADD_TRANSITION = enum.auto()
    REMOVE_TRANSITION = enum.auto()

    MUTATE_TRANSITION = enum.auto()
    MUTATE_INITIAL_STATE = enum.auto()

    MUTATE_INITIAL_DATA = enum.auto()
    MUTATE_INITIAL_POSITION = enum.auto()
    MUTATE_PADDING = enum.auto()

class TapeMachinePool:
    def __init__(self, max_population=10):
        self.max_population = max_population

        self.generation = 0
        self.population = []

        self.reset()

    def reset(self):
        self.generation = 0
        self.population = []

        base = TapeMachine()

        for i in range(self.max_population):
            self.population.append({
                'organism': base.copy(),
                'fitness': float('-inf')
            })

        print(f"Population reset")

    def cull(self, p=0.5):
        """
        Cull population based on fitnesses, keeping the top percentage of the
        population.
        """

        # Ensure that population is sufficiently randomly culled if all
        # organisms are equally fit
        random.shuffle(self.population)

        self.population.sort(key=lambda x: x['fitness'], reverse=True)

        sparecount = math.ceil(len(self.population) * p)

        # Ensure at least one survives
        sparecount = max(sparecount, 1)
        cullcount = len(self.population) - sparecount

        self.population = self.population[:sparecount]

    def _crossoverstr(self, a, b):
        if len(a) < 1:
            return b
        elif len(b) < 1:
            return a

        a = list(a)
        b = list(b)

        newname = ''

        while len(newname) < 1:
            x = random.randrange(len(a) + 1)
            y = random.randrange(len(b) + 1)
            newname = a[:x] + (b[-y:] if y > 0 else [])

        if random.randrange(4) == 0:
            # 1 in 4 chance to modify the new name by one character

            z = random.randrange(len(newname))

            if random.randrange(2) == 0:
                newname[z] = random.choice(a)
            else:
                newname[z] = random.choice(b)

        for i in range(len(newname)):
            if random.randrange(4) == 0:
                newname[i] = random.choice(string.ascii_letters)

        return ''.join(newname)

    def mutate(self, machine, mutationtype=None):
        if not isinstance(mutationtype, MutationType):
            mutationtype = random.choice(list(MutationType))

        charset = {
            'name': string.ascii_letters + string.digits,
            'condition': string.printable,
            'write': string.printable,
            'move': 'LR'
        }

        states = list(machine.reactions.keys())

        # If no states exist, create one
        if len(states) < 1:
            mutationtype = MutationType.ADD_STATE

        if mutationtype == MutationType.ADD_STATE:
            #print("!!! MUTATE NEW STATE !!!")

            newname = ''

            if len(states) < 2:
                # If there aren't enough states, just generate a random name
                newname = random.choice(charset['name'])
            else:
                # Generate a random name using crossover
                while len(newname) < 1 or newname in states:
                    a, b = random.sample(states, 2)
                    newname = self._crossoverstr(a, b)

            machine.reactions[newname] = None
        elif mutationtype == MutationType.DUPLICATE_STATE:
            state = random.choice(states)
            reactions = copy.deepcopy(machine.reactions[state])

            newname = ''

            if len(states) < 2:
                # If there aren't enough states, just generate a random name
                newname = random.choice(charset['name'])
            else:
                # Generate a random name using crossover
                samples = states.copy()
                samples.remove(state)

                while len(newname) < 1 or newname in states:
                    b = random.sample(states, 1)[0]
                    newname = self._crossoverstr(state, b)

            machine.reactions[newname] = copy.deepcopy(reactions)
        elif mutationtype == MutationType.REMOVE_STATE:
            state = random.choice(states)

            del machine.reactions[state]
        elif mutationtype == MutationType.ADD_TRANSITION:
            #print("!!! MUTATE NEW TRANSITION !!!")

            state = random.choice(states)

            if machine.reactions[state] is None:
                machine.reactions[state] = {}

            if None not in machine.reactions[state] and random.randrange(4) == 0:
                # 1 in 4 chance of being generic
                machine.reactions[state]['None'] = {}
            else:
                condition = ''

                while len(condition) < 1:
                    if len(condition) == 0:
                        condition = random.choice(charset['condition'])

                        continue

                    r = random.randrange(2)

                    if r == 0:
                        condition += random.choice(charset['condition'])
                    elif r == 1:
                        x = random.randrange(len(condition))

                        v = list(condition)
                        v[x] = random.choice(charset['condition'])

                        condition = ''.join(v)

                    condition = list(condition)

                    for existingcondition in machine.reactions[state].keys():
                        for c in existingcondition:
                            while c in condition:
                                condition.remove(c)

                    condition = str(condition)

                machine.reactions[state][condition] = {}
        elif mutationtype == MutationType.REMOVE_TRANSITION:
            state = random.choice(states)

            if machine.reactions[state] is None:
                return

            conditions = list(machine.reactions[state].keys())

            if conditions is None or len(conditions) < 1:
                return

            condition = random.choice(conditions)

            del machine.reactions[state][condition]
        elif mutationtype == MutationType.MUTATE_TRANSITION:
            #print("!!! MUTATE TRANSITION !!!")

            state = random.choice(states)

            if machine.reactions[state] is not None:
                conditions = list(machine.reactions[state].keys())

                if len(conditions) > 0:
                    condition = random.choice(conditions)

                    reaction = machine.reactions[state][condition]

                    action = random.choice([None, 'write', 'move', 'goto'])

                    if action == None:
                        # Alter condition
                        newcondition = list(condition)

                        r = random.randrange(3)

                        if r == 0 and len(newcondition) > 1 and condition != 'None':
                            # Remove random character
                            x = random.randrange(len(newcondition))
                            del newcondition[x]
                        elif r == 1:
                            newcondition = []
                        elif r == 2:
                            # Get a random character that doesn't exist in any
                            # other conditions
                            newcharacter = machine.padding

                            exists = True

                            # Max 50 attempts
                            for i in range(50):
                                exists = False

                                for othercondition in conditions:
                                    if newcharacter in othercondition:
                                        exists = True

                                        break

                                if exists:
                                    newcharacter = random.choice(charset['write'])

                            if exists:
                                return

                            if random.randrange(2) == 0 or condition == 'None':
                                # Append
                                newcondition.append(newcharacter)
                            else:
                                # Replace
                                x = random.randrange(len(newcondition))
                                newcondition[x] = newcharacter

                        if len(newcondition) > 0:
                            newcondition = ''.join(newcondition)
                        else:
                            newcondition = 'None'

                        del machine.reactions[state][condition]
                        machine.reactions[state][newcondition] = reaction
                    elif action == 'write':
                        r = random.randrange(2)

                        if r == 0 or 'write' not in reaction:
                            # Set to random character
                            reaction['write'] = random.choice(charset['write'])
                        elif r == 1:
                            # Modify slightly
                            character = ord(reaction['write'])
                            character += random.randint(-1, 1)
                            character = chr(character)

                            if character in charset['write']:
                                reaction['write'] = character
                    elif action == 'move':
                        reaction['move'] = random.choice(['L', 'R'])
                    elif action == 'goto':
                        reaction['goto'] = random.choice(states)
        elif mutationtype == MutationType.MUTATE_INITIAL_STATE:
            machine.initial_state = random.choice(states)
        elif mutationtype == MutationType.MUTATE_INITIAL_DATA:
            r = random.randrange(3)

            if r == 0:
                # Add character in random position
                x = random.randrange(len(machine.initial_data) + 1)

                machine.initial_data.insert(x, random.choice(charset['write']))

            if len(machine.initial_data) > 0:
                x = random.randrange(len(machine.initial_data))

                if r == 1:
                    # Remove character
                    del machine.initial_data[x]
                elif r == 2:
                    # Slightly modify character
                    character = ord(machine.initial_data[x])
                    character += random.randint(-1, 1)
                    character = chr(character)

                    if character in charset['write']:
                        machine.initial_data[x] = character
        elif mutationtype == MutationType.MUTATE_INITIAL_POSITION:
            machine.initial_position += random.randint(-1, 1)
        elif mutationtype == MutationType.MUTATE_PADDING:
            machine.padding = random.choice(charset['write'])
        else:
            raise ValueError("Unknown mutation type")

    def breed(self):
        organisms = []
        weights = []

        for entry in self.population:
            organisms.append(entry['organism'])
            weights.append(entry['fitness'])

        children = []

        while len(self.population) + len(children) < self.max_population:
            child = None

            r = random.randrange(2)

            if r == 0:
                # Breed by clone
                parent = random.choices(organisms, weights=weights, k=1)[0]
                child = parent.copy()
            elif r == 1:
                # Breed by crossover
                a, b = random.choices(organisms, weights=weights, k=2)

                astates = list(a.reactions.keys())
                bstates = list(b.reactions.keys())

                asegment = random.choices(astates, k=len(astates) // 2)
                bsegment = random.choices(bstates, k=len(bstates) // 2)

                childdescriptor = {
                    #'padding': random.choice([a.padding, b.padding]),
                    #'tape_size': random.choice([a.tape_size, b.tape_size]),
                    #'initial_position': random.choice([a.initial_position, b.initial_position]),
                    'initial_state': random.choice([a.initial_state, b.initial_state]),
                    'reactions': {}
                }

                for k in asegment:
                    # If a transition exists, randomly discard one
                    if k in childdescriptor['reactions']:
                        if random.randrange(2) == 0:
                            continue

                    childdescriptor['reactions'][k] = a.reactions[k]

                for k in bsegment:
                    # If a transition exists, randomly discard one
                    if k in childdescriptor['reactions']:
                        if random.randrange(2) == 0:
                            continue

                    childdescriptor['reactions'][k] = b.reactions[k]

                child = TapeMachine()
                child.load(childdescriptor)

                adata = ''.join(a.initial_data)
                bdata = ''.join(b.initial_data)
                cdata = self._crossoverstr(adata, bdata)

                child.initial_data = list(cdata)
                child.initial_position = random.choice([a.initial_position, b.initial_position])
                child.padding = random.choice([a.padding, b.padding])

            # Mutate child slightly
            mutationtypes = [
                MutationType.ADD_STATE,
                MutationType.DUPLICATE_STATE,
                MutationType.REMOVE_STATE,
                MutationType.ADD_TRANSITION,
                MutationType.REMOVE_TRANSITION,
                MutationType.MUTATE_TRANSITION,
                MutationType.MUTATE_INITIAL_STATE,
                MutationType.MUTATE_INITIAL_DATA,
                MutationType.MUTATE_INITIAL_POSITION,
                MutationType.MUTATE_PADDING
            ]

            mutationweights = [
                # ADD_STATE
                0.3,
                # DUPLICATE_STATE
                0.3,
                # REMOVE_STATE
                0.3,
                # ADD_TRANSITION
                2,
                # REMOVE_TRANSITION
                0.5,
                # MUTATE_TRANSITION
                2,
                # MUTATE_INITIAL_STATE
                0.1,
                # MUTATE_INITIAL_DATA
                1,
                # MUTATE_INITIAL_POSITION
                1,
                # MUTATE_PADDING
                0.5
            ]

            for i in range(random.randrange(10)):
                mutation = random.choices(mutationtypes, weights=mutationweights)
                self.mutate(child, mutationtype=mutation)

            children.append({
                'organism': child,
                'fitness': float('-inf')
            })

        self.population += children

    def generatetest(self):
        x = random.randint(0, 2 ** 8)
        y = random.randint(0, 2 ** 8)

        testinput = ''.join(random.choices(string.printable, k=128))
        #testinput = f""
        testoutput = f"Hello World!"

        return (testinput, testoutput)

    def getfitness(self, organism):
        """
        Customize the fitness function to whatever is necessary
        """

        def getsimilarity(target, readout):
            # Maximum difference of ord(c) for any pairs of c in string.printable
            # Precalculated
            maxdiff = 117

            score = 0

            for i in range(len(target)):
                if i > len(readout) - 1:
                    missed = len(target) - len(readout) + 1
                    score -= maxdiff * missed

                    break

                diff = ord(target[i]) - ord(readout[i])
                score -= abs(diff)

            return score

        max_iterations = 2048

        # Use a copy
        testorganism = organism.copy()

        # Load test settings

        digits = 16

        maxnumber = int(2 ** digits)

        #testinput = list(f"{x:b}+{y:b}")
        #expectedresult = x + y
        #expectedresultlength = len(f"{expectedresult:{digits}b}")

        testinput, targetreadout = self.generatetest()

        testorganism.tape_size = 1024
        testorganism.initial_data = list(testinput)

        # Reset machine to initial conditions before running
        testorganism.reset()

        iterations = 0
        fitness = 0

        halted = False

        try:
            # Simulate
            for i in range(max_iterations):
                halted, reason = testorganism.step()

                if halted:
                    break

            # Not including the initial one
            iterations = i
        except Exception as e:
            # Really discourage breaking any systems
            return float('-inf')

        # Get output
        readout = testorganism.readall()

        # Please halt
        if not halted or iterations == 0:
            return float('-inf')

        reactions = testorganism.reactions
        states = list(reactions.keys())
        statecount = len(states)

        emptystatecount = 0

        if statecount > 0:
            fitness += 10

        # Discourage conditions that do nothing, since it causes infinite loops
        for state in states:
            if reactions[state] is None:
                emptystatecount += 1

                continue

            conditions = list(reactions[state].keys())

            if len(conditions) < 1:
                emptystatecount += 1

                continue

            for condition in conditions:
                if reactions[state][condition] is None:
                    continue

                actions = list(reactions[state][condition].keys())
                actioncount = len(actions)

                if actioncount < 1:
                    fitness -= 1

        # Discourage many empty states
        fitness -= max(emptystatecount - 20, 0) * 1

        # Discourage large initial_data in original
        fitness -= len(organism.initial_data) * 5

        # Discourage large initial_position in original
        fitness -= abs(organism.initial_position) * 5

        # Maximum difference of ord(c) for any pairs of c in string.printable
        # Precalculated
        maxdiff = 117

        totalexact = 0
        totaldiff = 0

        if len(targetreadout) > len(readout):
            missed = len(targetreadout) - len(readout)
            totaldiff += missed * maxdiff

        for i in range(len(targetreadout) + 1):
            if i > len(readout) - 1:
                break

            # Encourage ending with padding
            if i == len(targetreadout):
                if i < len(readout) - 1:
                    if readout[i] == testorganism.padding:
                        fitness += 10

                break

            a = ord(targetreadout[i])
            b = ord(readout[i])
            diff = abs(a - b)

            totaldiff += diff

            if diff == 0:
                totalexact += 1

        fitness -= totaldiff
        fitness += totalexact * 10

        # Encourage using some targeted amount of states
        #targetstatecount = 5
        #fitness -= math.exp(abs(statecount - targetstatecount))

        #print(f"Readout: {readout}")
        #print(f"Fitness: {fitness}")
        #time.sleep(0.5)

        return fitness

    def step(self):
        """
        Step one generation
        """

        mutation_iterations = 10

        # Update fitness scores of population
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            entries = {}

            for entry in self.population:
                if entry['fitness'] > float('-inf'):
                    if random.randrange(2) == 0:
                        continue

                f = executor.submit(self.getfitness, entry['organism'])

                entries[f] = entry

            for f in concurrent.futures.as_completed(entries):
                entries[f]['fitness'] = f.result()

        self.cull()
        self.breed()

        self.generation += 1

    def printstats(self):
        bestfitness = 0

        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        bestfitness = self.population[0]['fitness']

        print(f"Generation {self.generation}")
        print(f"Best fitness: {bestfitness}")

def main():
    debug = 0

    if debug == 1:
        testpool = TapeMachinePool()

        testinput, testoutput = testpool.generatetest()

        machine = TapeMachine(file_path='fittest.yml')

        machine.initial_data = list(testinput)
        machine.reset()

        print(f"Initial state:\n{machine}")

        max_iterations = 2048

        halted = False

        try:
            for i in range(max_iterations):
                halted, reason = machine.step()

                print(machine)
                time.sleep(0.1)

                if halted:
                    print(f"Halted for reason: {reason}")

                    break
        except Exception as e:
            print(e)
            print("Did not halt")

        print(machine)

        print(f"Expected output: {testoutput}")

        print(f"Experimental fitness: {testpool.getfitness(machine)}")

        if not halted:
            print("Machine did not halt")
        else:
            print(f"Halted at iteration {i}")
    else:
        pool = TapeMachinePool(max_population=50)

        initialorganism = TapeMachine(file_path='fittest.yml')

        for e in pool.population:
            e['organism'] = initialorganism.copy()

        del initialorganism

        maxgenerations = 100_000

        for i in range(maxgenerations):
            pool.step()

            if (i + 1) % 100 == 0 and len(pool.population) > 0:
                pool.printstats()

            if (i + 1) % 500 == 0:
                pool.population.sort(key=lambda x: x['fitness'], reverse=True)
                machine = pool.population[0]['organism']
                machine.dumpf('fittest.yml')

        # Save fittest organism to file at end of iterations
        pool.population.sort(key=lambda x: x['fitness'], reverse=True)
        machine = pool.population[0]['organism']
        print(f"Current max fitness: {pool.population[0]['fitness']}")
        machine.dumpf('fittest.yml')

if __name__ == "__main__":
    main()
