import copy

import yaml

class TapeMachine:
    def __init__(self,
                 padding=' ', tape_size=1024,
                 initial_data='', initial_position=0, initial_state='main',
                 reactions={},
                 file_path=None):
        if file_path is not None:
            self.loadf(file_path)

            return

        self.padding = padding
        self.tape_size = tape_size
        self.initial_data = list(str(initial_data))
        self.initial_position = initial_position
        self.initial_state = initial_state

        self.reactions = reactions

        self._convertkeys()

        self.reset()

    def copy(self):
        duplicate = TapeMachine()

        duplicate.padding = self.padding
        duplicate.tape_size = self.tape_size
        duplicate.initial_data = copy.deepcopy(self.initial_data)
        duplicate.initial_position = self.initial_position
        duplicate.initial_state = self.initial_state

        duplicate.reactions = copy.deepcopy(self.reactions)

        duplicate.data = copy.deepcopy(self.data)
        duplicate.position = self.position
        duplicate.state = self.state

        return duplicate

    def load(self, descriptor):
        self.padding = descriptor.get('padding', ' ')
        self.tape_size = descriptor.get('tape_size', 1024)
        self.initial_data = list(str(descriptor.get('initial_data', '')))
        self.initial_position = descriptor.get('initial_position', 0)
        self.initial_state = descriptor.get('initial_state', 'main')

        self.reactions = descriptor.get('reactions', {})

        for state in self.reactions.keys():
            if self.reactions[state] is None:
                continue

            for condition in self.reactions[state].keys():
                if self.reactions[state][condition] is None:
                    continue

                for action in list(self.reactions[state][condition].keys()):
                    value = self.reactions[state][condition][action]

                    if not isinstance(value, str):
                        self.reactions[state][condition][action] = str(value)[:1]

        self._convertkeys()

        self.reset()

    def loadf(self, file_path):
        with open(file_path) as file:
            descriptor = yaml.safe_load(file.read())

        self.load(descriptor)

    def _convertkeys(self):
        for i in list(self.reactions.keys()):
            if not isinstance(i, str):
                new_i = str(i)
                value = self.reactions[i]

                del self.reactions[i]

                self.reactions[new_i] = value

                i = new_i

            if self.reactions[i] is not None:
                for j in list(self.reactions[i]):
                    if not isinstance(j, str):
                        new_j = str(j)
                        value = self.reactions[i][j]

                        del self.reactions[i][j]

                        self.reactions[i][new_j] = value

    def getdescriptor(self):
        return {
            'padding': self.padding,
            'tape_size': self.tape_size,
            'initial_data': ''.join(self.initial_data),
            'initial_position': self.initial_position,
            'initial_state': self.initial_state,
            'reactions': copy.deepcopy(self.reactions)
        }

    def dumps(self):
        return yaml.dump(self.getdescriptor())

    def dumpf(self, file_path):
        with open(file_path, "w") as file:
            file.write(self.dumps())

    def reset(self):
        """
        Reset to intial conditions
        """

        self.data = self.initial_data
        self.position = self.initial_position
        self.state = self.initial_state

    def read(self):
        if self.position < 0 or self.position > len(self.data) - 1:
            return self.padding[:1]

        return str(self.data[self.position])[:1]

    def write(self, value):
        if self.position > len(self.data) - 1:
            diff = self.position - len(self.data) + 1
            self.data += [self.padding[:1],] * diff
        elif self.position < 0:
            self.data = ([self.padding[:1],] * abs(self.position)) + self.data
            self.position = 0

        self.data[self.position] = str(value)[:1]

    def readall(self):
        data = ''.join(self.data)

        # Strip leading and trailing padding
        data = data.lstrip(self.padding[:1])
        data = data.rstrip(self.padding[:1])

        return data

    def move(self, direction):
        direction = str(direction).upper()[:1]

        nextposition = self.position

        if direction == 'L':
            nextposition -= 1
        elif direction == 'R':
            nextposition += 1
        else:
            nextposition += random.choice((-1, 1))

        # Enforce tape boundary

        newlength = 0

        if self.position < 0:
            newlength += abs(nextposition)
        else:
            newlength = max(nextposition + 1, len(self.data))

        if newlength > self.tape_size:
            return False

        self.position = nextposition

        return True

    def step(self):
        current_read = self.read()

        if self.state not in self.reactions:
            return True, "Unknown state"

        reactions = self.reactions[self.state]

        if reactions is None:
            return True, "Terminal state"

        matches = []
        genericmatch = None

        for k in reactions:
            if current_read in k:
                matches.append(k)
            elif k == 'None':
                genericmatch = k

        if len(matches) == 0 and genericmatch is not None:
            matches = [genericmatch]

        if len(matches) > 0:
            matches.sort()
            match = matches[0]
            reaction = reactions[match]

            write_value = reaction.get('write', None)
            move_direction = reaction.get('move', None)
            next_state = reaction.get('goto', None)

            if write_value is not None:
                self.write(write_value)

            if move_direction is not None:
                if not self.move(move_direction):
                    raise IndexError("Can't move past edge of tape")

            if next_state is not None:
                self.state = next_state

            return False, "Not halted"

        return True, "No condition matches input"

    def __repr__(self):
        tape_repr = f"<{''.join(self.data)}>"
        pos_repr = str(self.position)
        state_repr = str(self.state)

        return f"{tape_repr}\n{state_repr} at {pos_repr}"
