import random
import numpy as np
from golay import GolayCode
from rm import RMCode


def weight(x):
    """
    Функция вычисяет вес Хэмминга
    """
    return sum(x)


def add(x, y):
    """
    Функция вычисляет сумму двух векторов
    """
    return (x + y) % 2


def dot(x, y):
    """
    Функция реализует матричное умножение
    """
    return (x @ y) % 2


def generate_errors(n, number_of_errors):
    """
    Функция возвращает генератор ошибок для заданного числа ошибок в кодовом слове
    """
    if n == 1:
        yield [0]
        if number_of_errors == 1:
            yield [1]
    elif number_of_errors == 0:
        yield [0 for _ in range(n)]
    else:
        for err in generate_errors(n - 1, number_of_errors):
            yield err + [0]
        for err in generate_errors(n - 1, number_of_errors - 1):
            yield err + [1]


def create_table_of_syndromes(H, number_of_errors):
    """
    Функция формирует синдромы линейного кода (n, k, d)
    """
    errors = [np.array(error) for error in generate_errors(H.shape[0], number_of_errors)]
    syndromes = dot(np.array(errors), H)
    table_of_syndromes = {}
    for syndrome, error in zip(syndromes, errors):
        table_of_syndromes[tuple(syndrome)] = error
    return table_of_syndromes


def test_errors(G, H, syndromes, errors_count):
    """
    Функция для проведения теста на разном количестве ошибок
    """
    k, n = G.shape
    if errors_count > n:
        print(f'Количесво ошибок превышает длину кодовых слов ({errors_count} > {n})')
        return
    print('Тест количества ошибок:', errors_count)
    print('-------------------------')
    word = np.random.randint(0, 2, k)
    print('Слово:', word)
    message = dot(word, G)
    print('Сообщение:', message)
    error = np.zeros(n, dtype=int)
    possible_error_positions = [i for i in range(n)]
    for _ in range(errors_count):
        error[possible_error_positions.pop(random.randint(0, len(possible_error_positions) - 1))] = 1
    print('Вектор ошибки:', error)
    message_with_error = add(message, error)
    print('Сообщение с ошибками:', message_with_error)
    syndrome = dot(message_with_error, H)
    print('Синдром:', syndrome)
    calculated_message = message
    if weight(syndrome) == 0:
        print('Ошибок не обнаружено')
    else:
        syndrome_as_tuple = tuple(syndrome)
        if syndrome_as_tuple not in syndromes:
            print('Синдром не найден в таблице синдромов -> Невозможно исправить ошибку -> Повторный запрос сообщения')
            print(f'Код позволяет обнаружить количество ошибок: {errors_count}, но не позволяет их исправить')
            return
        print('Синдром найден в таблице синдромов -> Попытка исправить ошибку')
        calculated_error = syndromes[syndrome_as_tuple]
        print('Вычисленный вектор ошибки:', calculated_error,
              f'(количество найденных ошибок: {weight(calculated_error)})')
        calculated_message = add(message_with_error, calculated_error)
        print('Исправленное сообщение:', message)
    calculated_word = calculated_message[:k]
    print('Декодированное слово:', calculated_word)
    if weight(add(calculated_word, word)) == 0:
        print('Декодированное слово совпадает с исходным, код позволяет исправить количество ошибок:', errors_count)
    else:
        print('Декодированное слово не совпадает с исходным, код не позволяет найти или исправить количество ошибок:',
              errors_count)
    print()


def main():
    code = GolayCode()
    syndromes = create_table_of_syndromes(code.H, 1)
    test_errors(code.G, code.H, syndromes, 1)
    syndromes = create_table_of_syndromes(code.H, 2)
    test_errors(code.G, code.H, syndromes, 2)
    syndromes = create_table_of_syndromes(code.H, 3)
    test_errors(code.G, code.H, syndromes, 3)
    syndromes = create_table_of_syndromes(code.H, 4)
    test_errors(code.G, code.H, syndromes, 4)

    print("Тест кода Рида-Маллера")

    code = RMCode(1, 3)
    code.generate_G()
    code.generate_Hs()

    a = np.array([1, 1, 0, 0])
    print("Слово:", a)
    print("Правильное w:", code.encode(a))
    input_array = np.array([1, 0, 1, 0, 1, 0, 1, 1])
    output_array = code.decode(input_array)
    print('w с одной ошибкой:', input_array)
    print('Выходной массив:', output_array)
    print()

    a = np.array([1, 1, 0, 0])
    print("Слово:", a)
    print("Правильное w:", code.encode(a))
    input_array = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    output_array = code.decode(input_array)
    print('w с двумя ошибкой:', input_array)
    print('Выходной массив:', output_array)
    print()

    code = RMCode(1, 4)
    code.generate_G()
    code.generate_Hs()

    a = np.array([0, 0, 0, 1, 0])
    print("Слово:", a)
    print("Правильное w:", code.encode(a))
    input_array = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    output_array = code.decode(input_array)
    print('w с одной ошибкой:', input_array)
    print('Выходной массив:', output_array)
    print()

    a = np.array([0, 0, 0, 1, 0])
    print("Слово:", a)
    print("Правильное w:", code.encode(a))
    input_array = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    output_array = code.decode(input_array)
    print('w с двумя ошибками:', input_array)
    print('Выходной массив:', output_array)
    print()

    a = np.array([0, 0, 0, 1, 0])
    print("Слово:", a)
    print("Правильное w:", code.encode(a))
    input_array = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    output_array = code.decode(input_array)
    print('w с тремя ошибками:', input_array)
    print('Выходной массив:', output_array)
    print()

    a = np.array([0, 0, 0, 1, 0])
    print("Слово:", a)
    print("Правильное w:", code.encode(a))
    input_array = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    output_array = code.decode(input_array)
    print('w с четырьмя ошибками:', input_array)
    print('Выходной массив:', output_array)
    print()


if __name__ == '__main__':
    main()
