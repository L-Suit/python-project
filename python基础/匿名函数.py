def sort_test():
    students = [
        {'name': '小红', 'age': 18},
        {'name': '小明', 'age': 20},
        {'name': '小瑶', 'age': 18},
        {'name': '小田', 'age': 19}
    ]
    # sorted 默认升序，reverse默认是False
    print(sorted(students, key= lambda student: student['age']))

if __name__ == '__main__':
    sort_test()
