class RobotDummy:
    def __init__(self, mass, wheels, name):
        self.mass = mass
        self.wheels = wheels
        self.name = name
        
    def dummy_formula(self):
        return f"{self.mass} , {self.wheels} , {self.name}"
        

if __name__ == "__main__":
    mass = int(input("Enter mass, wheels, name :"))
    wheels, name = 0,0
    wheels = int(wheels)
    name = int(name)
    robot = RobotDummy(mass, wheels, name)
    print(robot.dummy_formula())