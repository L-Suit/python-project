class Dog():
    class_name='狗类'

    def __init__(self):
        print("init do")
        self.name='大黄'

    #实例方法
    def get_class_name(self):
        return Dog.class_name

    #类方法
    @classmethod
    def get_class_name(cls):
        return cls.class_name


dog1 = Dog()
print(dog1)
print(dog1.get_class_name())
print(Dog.get_class_name())
