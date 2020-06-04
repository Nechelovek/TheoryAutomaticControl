from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

variant = 1

class real_object:
    def __init__(self, variant):
        self._var = variant
        self._ctrl_fcn = real_object._default_control
        self.lin_par_1 = variant % 10 * 0.2
        self.lin_par_2 = ((32 - variant) % 9 + 0.1) * 2.5
        self.nonlin_par_1 = variant % 15 * 0.35
        self.nonlin_par_2 = variant % 12 * 0.45
        self.nonlin_fcns = [self.deadZone, self.saturation, self.relay]
        self.nonlin_names = ['deadZone', 'saturation', 'relay']
        self.nonlin_type = variant % 3
        self._params = [self.lin_par_1, self.lin_par_2, self.nonlin_par_1, self.nonlin_par_2]
        print(self.lin_par_1, self.lin_par_2, self.nonlin_par_1, self.nonlin_par_2)

    def deadZone(self, x, p1, p2):
        if np.abs(x) < p1:
            x = 0
        elif x > 0:
            x = x - p1
        elif x < 0:
            x = x + p2
        return x

    def saturation(self, x, p1, p2):
        if x > p1:
            x = p1
        elif x < -p2:
            x = -p2
        return x

    def relay(self, x, p1, p2):
        if x > 0:
            return p1
        else:
            return -p2

    def _ode(self, x, t, k):
        '''
        Функция принимает на вход вектор переменных состояния и реализует по сути систему в форме Коши
        x -- текущий вектор переменных состояния
        t -- текущее время
        k -- значения параметров
        '''
        y = x
        u = self._get_u(x, t)
        lin_par_1, lin_par_2, nonlin_par_1, nonlin_par_2 = k

        dydt = (lin_par_1 * self.nonlin_fcns[self.nonlin_type](u, nonlin_par_1, nonlin_par_2) - x) / lin_par_2
        return dydt

    def _default_control(x, t):
        """
        Управление по умолчанию. Нулевой вход
        """
        return 0

    def _get_u(self, x, t):
        """
        Получить значение управления при значениях переменных состояния x и времени t
        """
        return self._ctrl_fcn(x, t)

    def set_u_fcn(self, new_u):
        """
        Установить новую функцию управления
        формат функции: fcn(x, t)
        """
        self._ctrl_fcn = new_u

    def calcODE(self, y0, ts=800, nt=1001):
        """
        Вспомогательная функция для получения решения систему ДУ, "Проведение эксперимента" с заданным воздействием
        """
        y0 = [y0, ]
        t = np.linspace(0, ts, nt)
        args = (self._params,)
        sol = odeint(self._ode, y0, t, args)
        return sol, t

    def getODE(self):

        #Получить "идеальную" модель объекта без параметров

        return self._ode

    def get_nonlinear_element_type(self):
        return self.nonlin_names[self.nonlin_type]

obj = real_object(variant)
y0 = 1
sol, t = obj.calcODE(y0)
plt.plot(t, sol)
plt.grid()
plt.show()

def exp(sig):
    # Создаем экземпляр класса объекта
    obj = real_object(variant)

    # Задаем функцию управления
    obj.set_u_fcn(sig)

    # Задаем начальные условия
    y0 = 1

    # Проводим эксперимент
    sol, t = obj.calcODE(y0)

    # И строим управляющее воздействие. В случае, если функция управления -- скалярная, то ее стоит векторизовать.
    u = sig(0, t)
    plt.plot(t, u)
    plt.plot(t, sol)
    plt.grid()
    plt.show()

def monoharm_u(x, t):
    return 1*np.sin(t * 0.025 * np.pi)

exp(monoharm_u)

def monoharm_e(x, t):
    return 1*np.sin(t * 0.025 * np.pi)

#exp(monoharm_e)


def step_function(x, t):
    if(t >= 0):
        return 1
    else:
        return 0
step_function = np.vectorize(step_function, otypes = [np.float])

exp(step_function)

def impulse(x,t): #Aproksimatiya
  if t < 0.01:
    return 100
  else:
    return 0
impulse = np.vectorize(impulse,  otypes = [np.float])

exp(impulse)

def square(x, t):
    return 0.5 * signal.square(t * 0.025 * np.pi)

exp(square)

def whitenoise(x, t):
  return np.random.normal(1,1)
whitenoise = np.vectorize(whitenoise, otypes=[np.float])

exp(whitenoise)

#Chapter 2
from scipy import optimize
from scipy import integrate, interpolate


class parameter_estimator():
    def __init__(self, experiments, f):
        """
        experiments -- список кортежей с данными экспериментов в формате [x_data, y_data] (вход, выход)
        f -- функция, реализующая дифференциальное уравнение модели
        """
        self._experiments = experiments
        self._f = f
        # Предполагаем, что все переменные состояния наблюдаемые, однако в общем случае это не так
        x_data, y_data = experiments[0]
        self.n_observed = 1  # y_data.shape[1]

    def my_ls_func(self, x, teta):
        """
        Определение функции, возвращающей значения решения ДУ в
        процессе оценки параметров
        x заданные (временные) точки, где известно решение
        (экспериментальные данные)
        teta -- массив с текущим значением оцениваемых параметров.
        Первые self._y0_len элементов -- начальные условия,
        остальные -- параметры ДУ
        """
        # Для передачи функуии используем ламбда-выражение с подставленными
        # параметрами
        # Вычислим значения дифференциального уравления в точках "x"
        r = integrate.odeint(lambda y, t: self._f(y, t, teta[self._y0_len:]),
                             teta[0:self._y0_len], x)
        # Возвращаем только наблюдаемые переменные
        return r[:, 0:self.n_observed]

    def estimate_ode(self, y0, guess):
        """
        Произвести оценку параметров дифференциального уравнения с заданными
        начальными значениями параметров:
            y0 -- начальные условия ДУ
            guess -- параметры ДУ
        """
        # Сохраняем число начальных условий
        self._y0_len = len(y0)
        # Создаем вектор оцениваемых параметров,
        # включающий в себя начальные условия
        est_values = np.concatenate((y0, guess))
        c = self.estimate_param(est_values)
        # В возвращаемом значении разделяем начальные условия и параметры
        return c[self._y0_len:], c[0:self._y0_len]

    def f_resid(self, p):
        """
        Функция для передачи в optimize.leastsq
        При дальнейших вычислениях значения, возвращаемые этой функцией,
        будут возведены в квадрат и просуммированы.

        """
        delta = []
        # Получаем ошибку для всех экспериментов при заданных параметрах модели
        for data in self._experiments:
            x_data, y_data = data
            d = y_data - self.my_ls_func(x_data, p)
            d = d.flatten()
            delta.append(d)
        delta = np.array(delta)

        return delta.flatten()  # Преобразуем в одномерный массив

    def calcODE(self, args, y0, x0=0, xEnd=10, nt=1001):
        """
        Служебная функция для решения ДУ
        """
        t = np.linspace(x0, xEnd, nt)
        sol = odeint(self._f, y0, t, args)
        return sol, t

    def estimate_param(self, guess):
        """
        Произвести оценку параметров ДУ
            guess -- параметры ДУ
        """
        self._est_values = guess
        # Решить оптимизационную задачу - решение в переменной c
        res = optimize.least_squares(self.f_resid, self._est_values)
        return res.x

def ideal_model(signal):
    guess = [0.2, 10.25, 0.35, 0.45]  # Начальные значения для параматров системы
    y0 = [1, ]  # Стартовые начальные значения для системы ДУ

    obj = real_object(variant)
    obj.set_u_fcn(signal)
    sol, t = obj.calcODE(y0[0])

    estimator = parameter_estimator([[t, sol], ], obj.getODE())
    est_par = estimator.estimate_ode(y0, guess)
    print("Estimated parameter: {}".format(est_par[0]))
    print("Estimated initial condition: {}".format(est_par[1]))

     # Строим графики входа и выходов системы
    y0 = est_par[1]
    args = (est_par[0],)
    sol_ideal = odeint(obj.getODE(), y0, t, args)

    # Строим графики входа и выходов системы
    plt.plot(t, sol_ideal)
    plt.plot(t, sol)
    plt.grid()
    plt.show()

ideal_model(monoharm_u)
#ideal_model(monoharm_e)
ideal_model(step_function)
ideal_model(impulse)
ideal_model(square)
ideal_model(whitenoise)

def ode_lin(x, t, k):
    """
    Функция, рализующая систему ДУ маятника с трением
    """
    y = x
    K = k[0]
    T = k[1]
    u = monoharm_u(0, t)
    dydt = (K*u-y)/T
    return dydt

def lineary_model(signal):
    guess = [0.2, 10.25, 0.35, 0.45]  # Начальные значения для параматров системы
    y0 = [0,]  # Стартовые начальные значения для системы ДУ

    obj = real_object(variant)
    obj.set_u_fcn(signal)
    sol, t = obj.calcODE(y0[0])

    estimator = parameter_estimator([[t, sol],], ode_lin)
    est_par = estimator.estimate_ode(y0, guess)
    print("Estimated parameter: {}".format(est_par[0]))
    print("Estimated initial condition: {}".format(est_par[1]))

    y0 = est_par[1]
    args = (est_par[0],)
    sol_lin = odeint(ode_lin, y0, t, args)

    plt.plot(t, sol_lin)
    plt.plot(t, sol)
    plt.grid()
    plt.show()

#lineary_model(monoharm_e)
lineary_model(monoharm_u)
lineary_model(step_function)
lineary_model(impulse)
lineary_model(square)
lineary_model(whitenoise)

# print(obj.get_nonlinear_element_type())

def ode_non_lineary(x, t, k):
    obj = real_object(variant)
    dydt = obj.saturation(x, k[0], k[1])
    return dydt

def non_lineary_model(signal):
    guess = [1, 2]  # Начальные значения для параматров системы
    y0 = [1, ]  # Стартовые начальные значения для системы ДУ

    obj = real_object(variant)
    obj.set_u_fcn(signal)
    sol, t = obj.calcODE(y0[0])

    estimator = parameter_estimator([[t, sol],], ode_non_lineary)
    est_par = estimator.estimate_ode(y0, guess) #Error
    print("Estimated parameter: {}".format(est_par[0]))
    print("Estimated initial condition: {}".format(est_par[1]))

    y0 = est_par[1]
    args = (est_par[0],)
    sol_nonlin = odeint(ode_non_lineary, y0, t, args, mxstep= 1000)

    plt.plot(t, sol_nonlin, label = "Nonlinear")
    plt.plot(t, sol, label = "Real")
    plt.grid()
    plt.legend()
    plt.show()

# non_lineary_model(monoharm_u)
# # non_lineary_model(monoharm_e)
# # non_lineary_model(step_function)
# # non_lineary_model(impulse)
# # non_lineary_model(square)
# # non_lineary_model(whitenoise)
#
def series2dataset(data, seq_len):
    """
    Преобразование временной последовательнсти к формату датасета
    Шаг дискретизации должен быть постоянным
    """
    dataset = []
    for i in range(data.shape[0]-seq_len):
        r = np.copy(data[i:i+seq_len])
        dataset.append(r)
    return np.array(dataset)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))

values = monoharm_u(0, t)
model_order = 5
# Этой командой получается массив обучающих последовательностей с одного
# эксперимента, для работы с несколькими экспериментами, можно получить
# последовательно массивы отдельно по каждому эксперименту, а затем объединить
# массивы
x_values = series2dataset(values, model_order)
x_values = np.expand_dims(x_values,2)
print(x_values.shape)

# Разделим на тестовую и обучающие выборки
# В случае использования нескольких экспериментов (>5), в качестве тестового
# лучше взять один из экспериментов целиком
n_train_samples = int(x_values.shape[0] * 0.7)

train_X = x_values[:n_train_samples, :]
test_X = x_values[n_train_samples:, :]

y_values = scaler.fit_transform(sol)
y_values = y_values[model_order:]
train_y = y_values[:n_train_samples]
test_y = y_values[n_train_samples:]

print("Shape of train is {}, {}, shape of test is {}, {}".format(train_X.shape,
                                                                 train_y.shape,
                                                                 test_X.shape,
                                                                 test_y.shape))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
history = model.fit(train_X,
                    train_y,
                    epochs=100,
                    batch_size=72,
                    validation_data=(test_X, test_y),
                    verbose=1,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt

#делаем предсказание
yhat = model.predict(test_X)

# обратное масштабирование для прогноза
inv_yhat = yhat #scaler.inverse_transform(yhat)
#inv_yhat = inv_yhat[:,0]
# обратное масштабирование для фактического
inv_y = test_y #scaler.inverse_transform(test_y)
#inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# Соединяем обратно входные данные
X = np.concatenate((train_X, test_X), axis=0)
Y_real = np.concatenate((train_y, test_y), axis=0)
#делаем предсказание
Y = model.predict(X)

print(X.shape, Y.shape)

plt.plot(Y)
plt.plot(Y_real)