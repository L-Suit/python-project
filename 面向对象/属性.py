

class Dog():
    def play(self):
        print(f'self:{id(self)}')
        print(f'小狗{self.name}在快乐的拆家')

dog = Dog()
dog1= Dog()
dog.name='二哈'
dog1.name = '金毛'
print({id(dog)})
print(dog)
dog.play()
dog1.play()