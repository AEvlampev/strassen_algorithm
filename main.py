"""Работа студента группы ИВТ-11БО Евлампьева Александра"""


from random import randint

# Объявляем глобальные переменные для подсчёта количества элементарных операций сложения и умножения
cnt_of_additions, cnt_of_multiplications = 0, 0


def input_of_k(inp: str) -> int:
    """
    Функция, проверяющая корректность ввода параметра k.
    :param inp: Строка, из которой считывается значение k.
    :return: Значение k.
    """
    param_k = 0
    # Пытаемся преобразовать строку в число, в случае неудачи выбрасываем код ошибки
    try:
        param_k = int(inp)
    except ValueError:
        return -1

    # Также выбрасываем код ошибки, если k - неположительное
    if param_k <= 0:
        return -1

    return param_k


def input_sizes(inp: str) -> tuple:
    """
    Функция, проверяющая корректность ввода размеров матрицы
    :param inp: Строка, из которой должны быть считаны размеры матрицы.
    :return: Кортеж, в котором содержится список с размерами матрицы и флаг успешности считывания размеров.
    """
    inp = inp.split()
    res = []

    # Если после разделения по пробелам/пустым местам окажется, что там было не два элемента, то выбрасываем ошибку
    if len(inp) != 2:
        return False, []
    for s in inp:
        # Пробуем преобразовать элемент в число; если не получается, то выбрасываем ошибку
        try:
            size = int(s)
        except ValueError:
            return False, []

        # Если один из размеров - отрицательный, то выбрасываем ошибку.
        if size <= 0:
            return False, []

        res += [size]

    return True, res


def input_elements(inp: str, len_of_row: int) -> tuple:
    """
    Функция, проверяющая корректность ввода строки матрицы
    :param inp: Строка, в которой содержатся значения элементов n-ой строки матрицы Х.
    :param len_of_row: Количество элементов в строке матрицы Х, которое пользователь ввёл ранее.
    :return: Кортеж, в котором содержится список с элементами n-ой строки матрицы Х и флаг успешности считывания
    """
    elem = None
    inp = inp.split()
    res = []

    for e in inp:
        # Если элемент - не целое число, выбрасываем False
        try:
            elem = int(e)
        except ValueError:
            return False, []

        res += [elem]

    # Если после преобразований в строке окажется меньше элементов, чем должно, то выбрасываем False
    if len(res) != len_of_row:
        return False, []

    return True, res


def classic_matrix_multiplication(first_matrix: list, second_matrix: list) -> list:
    """
    Функция, выполняющая умножение двух матриц по определению
    :param first_matrix: Первый множитель
    :param second_matrix: Второй множитель
    :return: Результирующая матрица
    """
    global cnt_of_multiplications, cnt_of_additions

    rows_of_first_matrix, cols_of_first_matrix, cols_of_second_matrix = len(first_matrix), \
        len(first_matrix[0]), len(second_matrix[0])
    result_matrix = [[[] for col in range(cols_of_second_matrix)] for row in range(rows_of_first_matrix)]

    for i in range(rows_of_first_matrix):
        for j in range(cols_of_second_matrix):
            for k in range(cols_of_first_matrix):
                result_matrix[i][j] += [first_matrix[i][k] * second_matrix[k][j]]
                cnt_of_multiplications += 1

    cnt_of_elements_in_cell = len(result_matrix[0][0])
    result_matrix = [[sum(cell) for cell in row] for row in result_matrix]

    # Отдельно выполняем подсчёт элементарных операций сложения
    cnt_of_additions += ((cnt_of_elements_in_cell - 1) * len(result_matrix) * len(result_matrix[0]))

    return result_matrix


def matrix_addition(first_matrix: list, second_matrix: list) -> list:
    """
    Функция, выполняющая сложение двух матриц
    :param first_matrix: Первое слагаемое
    :param second_matrix: Второе слагаемое
    :return: Результирующая матрица
    """
    global cnt_of_additions

    # Получаем количество столбцов и строк и создаём результирующую матрицу
    rows, cols = len(first_matrix), len(first_matrix[0])
    result_matrix = [[0 for col in range(cols)] for row in range(rows)]

    for row in range(rows):
        for col in range(cols):
            result_matrix[row][col] = (first_matrix[row][col] + second_matrix[row][col])
            cnt_of_additions += 1

    return result_matrix


def matrix_subtraction(first_matrix: list, second_matrix: list) -> list:
    """
    Функция, выполняющая вычитание второй матрицы из первой
    :param first_matrix: Уменьшаемая матрица
    :param second_matrix: Вычитаемая матрица
    :return: Результирующая матрица
    """
    global cnt_of_additions

    # Получаем количество столбцов и строк и создаём результирующую матрицу
    rows, cols = len(first_matrix), len(first_matrix[0])
    result_matrix = [[0 for col in range(cols)] for row in range(rows)]

    for row in range(rows):
        for col in range(cols):
            result_matrix[row][col] = (first_matrix[row][col] - second_matrix[row][col])
            cnt_of_additions += 1

    return result_matrix


def input_of_matrix(sizes: list, m_of_input: str, name_of_matrix: str) -> list:
    """
    Функция, реализующая построчный ввод матрицы
    :param sizes: Список, содержащий размеры вводимой матрицы.
    :param m_of_input: Способ ввода матрицы: вручную или рандомно
    :param name_of_matrix: Строка, содержащая название матрицы, для сообщений пользователю
    :return: Целевая матрица
    """
    matrix = []
    # Оставим вариативность на случай, если пользователь воспримет всё буквально
    if m_of_input == '0' or m_of_input == '"0"':
        for i in range(sizes[0]):
            # Осуществляем построчный ввод и передаём введённую строку на проверку корректности введённых значений
            print(f'Введите элементы {i + 1}-ой строки матрицы {name_of_matrix}'
                  f' ({sizes[0]} x {sizes[1]}) через пробел:')

            elements = input()
            inp_is_correct_row, row = input_elements(elements, sizes[1])

            # Если строка некорректна, то выбрасываем ошибку
            if not inp_is_correct_row:
                # incorrect_input_error_handler()
                return [False]

            matrix += [row]

        return matrix
    else:
        # Генерируем рандомную матрицу с целевыми размерами
        return [[randint(-9, 9) for j in range(sizes[1])] for i in range(sizes[0])]


def matrix_rows_attachment(matrix: list, delta: int) -> list:
    """
    Функция, отвечающая за достраивание матрицы нулевыми строками.
    :param matrix: Исходная матрица
    :param delta: Количество строк, которые необходимо достроить.
    :return: Достроенная матрица
    """

    # Если достраивать матрицу не надо, то возвращаем исходную матрицу
    if delta == 0:
        return matrix

    new_matrix = matrix

    # Достраиваем матрицу нулевыми строками
    len_of_row = len(matrix[0])
    for i in range(delta):
        new_matrix.append([0 for _ in range(len_of_row)])

    return new_matrix


def matrix_cols_attachment(matrix: list, delta: int) -> list:
    """
    Функция, отвечающая за достраивание матрицы нулевыми столбцами.
    :param matrix: Исходная матрица
    :param delta: Количество столбцов, которые необходимо достроить.
    :return: Достроенная матрица
    """

    # Если достраивать матрицу не надо, то возвращаем исходную матрицу
    if delta == 0:
        return matrix

    new_matrix = matrix

    rows = len(matrix)

    # Достраиваем матрицу нулевыми строками
    for row in range(rows):
        for _ in range(delta):
            new_matrix[row] += [0]

    return new_matrix


def matrix_attachment(first_matrix: list, second_matrix: list, k: int) -> tuple:
    """
    Функция, отвечающая за достраивание матрицы нулевыми элементами до размеров, которые позволят разделить её на блоки.
    Достраиваются сразу две матрицы, чтобы гарантировать корректность операций, которые потом будут производиться
    над ними.
    :param first_matrix: Первая матрица
    :param second_matrix: Первая матрица
    :param k: Параметр, отвечающий за то, матрица какого размера должна быть умножена не по Алгоритму Штрассена, а по
    определению
    :return: Кортеж с двумя достроенными матрицами
    """
    # Получаем максимальный из размеров двух матриц (строки х столбцы), до которого мы будем расширять матрицы
    lst_of_sizes = [len(first_matrix), len(first_matrix[0]), len(second_matrix), len(second_matrix[0])]
    max_size = max(lst_of_sizes)

    # Увеличиваем этот размер до тех пор, пока мы не сможем поделить матрицы на блоки так, чтобы они были матрицами
    # размера k x k или меньше
    flag = True
    while flag:
        max_size_copy = max_size
        flag_div_2 = True
        while max_size_copy > k:
            if max_size_copy % 2 == 0:
                max_size_copy //= 2
            else:
                flag_div_2 = False
                break
        if flag_div_2:
            break
        else:
            max_size += 1

    # Достраиваем матрицы
    new_first_matrix = matrix_rows_attachment(first_matrix, max_size - lst_of_sizes[0])
    new_second_matrix = matrix_rows_attachment(second_matrix, max_size - lst_of_sizes[2])

    new_first_matrix = matrix_cols_attachment(new_first_matrix, max_size - lst_of_sizes[1])
    new_second_matrix = matrix_cols_attachment(new_second_matrix, max_size - lst_of_sizes[3])

    return new_first_matrix, new_second_matrix


def matrix_separation(matrix: list) -> list:
    """
    Функция, реализующая разделение матрицы на блоки
    :param matrix: Исходная матрица
    :return: Блочная матрица по исходной
    """
    # Считаем половину от длины строки матрицы
    n = len(matrix)
    half = n // 2

    # Список границ индексов строк/столбцов, которые попадут в тот или иной блок
    limits = [((0, half), (0, half)), ((0, half), (half, n)), ((half, n), (0, half)), ((half, n), (half, n))]

    blocks = []

    # Разделяем матрицу
    for limit in limits:
        block = []
        for row in range(limit[0][0], limit[0][1]):
            block_row = []
            for col in range(limit[1][0], limit[1][1]):
                block_row += [matrix[row][col]]
            block += [block_row]
        blocks += [block]

    return blocks


def matrix_unification(block_matrix: list) -> list:
    """
    Функция, собирающая блочную матрицу в просто матрицу
    :param block_matrix: Блочная матрица.
    :return: Матрица, собранная из блочной.
    """
    block11, block12, block21, block22 = block_matrix[0], block_matrix[1], block_matrix[2], block_matrix[3]

    # Собираем верхнюю и нижнюю половины матрицы по отдельности и потом объединяем их в конечную
    upper_matrix = block11
    for row in range(len(upper_matrix)):
        for col in range(len(upper_matrix)):
            upper_matrix[row].append(block12[row][col])

    lower_matrix = block21
    for row in range(len(lower_matrix)):
        for col in range(len(lower_matrix)):
            lower_matrix[row].append(block22[row][col])

    result_matrix = upper_matrix + lower_matrix

    return result_matrix


def strassen_matrix_multiplication(first_matrix: list, second_matrix: list, k: int) -> list:
    """
    Функция, выполняющая умножение матриц по Штрассену.
    :param first_matrix: Первый множитель
    :param second_matrix: Второй множитель
    :param k: Параметр, отвечающий за то, матрица какого размера должна быть умножена не по Алгоритму Штрассена, а по
    определению
    :return: Результирующая матрица
    """
    len_first_matrix = len(first_matrix)

    # Если размер поступивших квадратных матриц меньше k, то выполняем умножение по определению, в ином случае -
    # спускаемся в рекурсию
    if len_first_matrix <= k:
        return classic_matrix_multiplication(first_matrix, second_matrix)
    else:
        # Делим матрицы на блочные
        first_block_matrix = matrix_separation(first_matrix)
        second_block_matrix = matrix_separation(second_matrix)

        # Рассчитываем вспомогательные значения для вычисления результирующей матрицы, используя рекурсию

        m1 = strassen_matrix_multiplication(matrix_addition(first_block_matrix[0], first_block_matrix[3]),
                                            matrix_addition(second_block_matrix[0], second_block_matrix[3]), k)

        m2 = strassen_matrix_multiplication(matrix_addition(first_block_matrix[2],
                                                            first_block_matrix[3]), second_block_matrix[0], k)

        m3 = strassen_matrix_multiplication(first_block_matrix[0],
                                            matrix_subtraction(second_block_matrix[1], second_block_matrix[3]), k)

        m4 = strassen_matrix_multiplication(first_block_matrix[3],
                                            matrix_subtraction(second_block_matrix[2], second_block_matrix[0]), k)

        m5 = strassen_matrix_multiplication(matrix_addition(first_block_matrix[0], first_block_matrix[1]),
                                            second_block_matrix[3], k)

        m6 = strassen_matrix_multiplication(matrix_subtraction(first_block_matrix[2], first_block_matrix[0]),
                                            matrix_addition(second_block_matrix[0], second_block_matrix[1]), k)

        m7 = strassen_matrix_multiplication(matrix_subtraction(first_block_matrix[1], first_block_matrix[3]),
                                            matrix_addition(second_block_matrix[2], second_block_matrix[3]), k)

        block11 = matrix_addition(matrix_subtraction(matrix_addition(m1, m4), m5), m7)

        block12 = matrix_addition(m3, m5)

        block21 = matrix_addition(m2, m4)

        block22 = matrix_addition(matrix_subtraction(m1, m2), matrix_addition(m3, m6))

        # Собираем из блочной матрицы результирующую
        result_matrix = matrix_unification([block11, block12, block21, block22])

        return result_matrix


def matrix_print(matrix: list, rows: int, cols: int, message: str) -> None:
    """
    Функция, отвечающая за вывод матрицы в консоль
    :param matrix: Список, содержащий матрицу, которую необходимо вывести.
    :param rows: Количество строк матрицы, которые необходимо вывести.
    :param cols: Количество столбцов матрицы, которые необходимо вывести.
    :param message: Строка, содержащая вспомогательное сообщение для пользователя.
    :return: Функция ничего не возвращает, поэтому None
    """
    if rows > 30 or cols > 30:
        print('Матрица слишком большая для вывода в консоль. Вывод осуществлён не будет.')
    else:
        # Выводим вспомогательное сообщение для пользователя
        print(message)

        # Осуществляем красивый вывод матрицы в консоль
        lst = [[str(matrix[row][col]) for col in range(cols)] for row in range(rows)]
        lens = [max(map(len, col)) for col in zip(*lst)]
        scheme = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [scheme.format(*row) for row in lst]
        print('\n'.join(table))


# Считываем размеры матрицы А
input_of_sizes_of_A = input("Введите размер матрицы A (m x n) через пробел:\n")

# Проверяем корректность введённых значений размеров матрицы
inp_is_correct_A, sizes_of_A = input_sizes(input_of_sizes_of_A)

# Если значения некорректны, то выбрасываем ошибку
if not inp_is_correct_A:
    print("Некорректный ввод. Работа программы некорректна завершена.")
    exit(-1)

# Аналогично для матрицы B
input_of_sizes_of_B = input("Введите размер матрицы B (n x p) через пробел:\n")

inp_is_correct_B, sizes_of_B = input_sizes(input_of_sizes_of_B)

if not inp_is_correct_B:
    print("Некорректный ввод. Работа программы некорректна завершена.")
    exit(-1)

# Если количество столбцов матрицы А не совпадает с количеством строк матрицы В, то выбрасываем ошибку
if sizes_of_A[1] != sizes_of_B[0]:
    print("Умножение матриц таких размеров невозможно. Работа программы некорректно завершена.")
    exit(-1)

# Считываем значение параметра k
k = input("Введите значение параметра k:\n")

# Проверяем значение параметра k
k = input_of_k(k)

if k == -1:
    print("Некорректный ввод матрицы. Работа программы некорректна завершена.")
    exit(-1)


m_of_input = input(
    'Если вы хотите вводить матрицу вручную, то введите "0".\nЕсли вы хотите, '
    'чтобы матрицы сгенерировались произвольно,'
    ' то просто введите произвольную последовательность символов.\n')

flag_output = True
if sizes_of_A[0] > 30 or sizes_of_A[1] > 30 or sizes_of_B[0] > 30 or sizes_of_B[1] > 30:
    print('')
    print('Размер входных матриц слишком велик для вывода. Вывод входных и конечных матриц отключен.')
    flag_output = False

# Осуществляем ввод матриц А и В
matrix_A = input_of_matrix(sizes_of_A, m_of_input, 'A')

if matrix_A == [False]:
    print("Некорректный ввод матрицы. Работа программы некорректна завершена.")
    exit(-1)

matrix_B = input_of_matrix(sizes_of_B, m_of_input, 'B')

if matrix_A == [False]:
    print("Некорректный ввод матрицы. Работа программы некорректна завершена.")
    exit(-1)

print('')

# Если матрицы слишком большие для вывода, то не выводим их
if flag_output:
    matrix_print(matrix_A, len(matrix_A), len(matrix_A[0]), 'Матрица А')
    print('')
    matrix_print(matrix_B, len(matrix_B), len(matrix_B[0]), 'Матрица B')
    print('')

# Умножаем две матрицы по определению
classic_res_matrix = classic_matrix_multiplication(matrix_A, matrix_B)

# Если матрицы слишком большие для вывода, то не выводим их
if flag_output:
    matrix_print(classic_res_matrix, sizes_of_A[0], sizes_of_B[1], 'Произведение матриц по определению')

# Выводим количество элементарных операций
print('Число элементарных операций умножения для умножения по определению')
print(cnt_of_multiplications)

print('Число элементарных операций сложения для умножения по определению')
print(cnt_of_additions)

print('')

# Обнуляем счётчики
cnt_of_additions, cnt_of_multiplications = 0, 0

# Дополняем матрицы для их дальнейшего разделения на блочные
matrix_A, matrix_B = matrix_attachment(matrix_A, matrix_B, k)

# Умножаем две матрицы по алгоритму Штрассена
strassen_res_matrix = strassen_matrix_multiplication(matrix_A, matrix_B, k)

# Если матрицы слишком большие для вывода, то не выводим их
if flag_output:
    matrix_print(strassen_res_matrix, sizes_of_A[0], sizes_of_B[1], 'Произведение матриц по алгоритму Штрассена')

# Выводим количество элементарных операций
print('Число элементарных операций умножения для умножения по алгоритму Штрассена')
print(cnt_of_multiplications)

print('Число элементарных операций сложения для умножения по алгоритму Штрассена')
print(cnt_of_additions)
