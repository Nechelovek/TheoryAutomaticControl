import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

variant = 1

test_signal_duration = 100
dt = 0.01
test_sig_ampl = 1 + variant * 0.1
test_sig_freq = 1 + variant * 3.5
non_lin_param_1 = 0.5 + variant * 0.1
lin_param_k = 0.5 + variant * 0.3
lin_param_T = 0.1 + variant * 0.2

print("Вариант номер {}".format(variant))
print("Период дискретизации сигнала: {:.2} с".format(dt))
print("Амплитуда тестового сигнала: {:.2}".format(test_sig_ampl))
print("Частота тестового сигнала: {:.2} Гц".format(test_sig_freq))
print("Длительность тестового сигнала: {} с".format(test_signal_duration))
print("Параметр нелинейностей 1: {:.2}".format(non_lin_param_1))
print("Коэффициент усиления линейного звена: {:.2}".format(lin_param_k))
print("Постоянная времени линейного звена: {:.2}".format(lin_param_T))

t = np.arange(0, test_signal_duration, dt) # Создаем вектор времени от 0 до 100 секунд с чатсотой дискретизации 0,01

print("Размерность массива: {}".format(t.shape))
print("Содержимое массива: {}".format(t))

# Test signal sin
sig_sin = test_sig_ampl * np.sin(t * 2 * np.pi * test_sig_freq)

print("Размерность сигнала: {}".format(sig_sin.shape))
print("Содержимое массива сигнала: {}".format(sig_sin))

plt.plot(t[0 : 100], sig_sin[0 : 100])

# Spectre test sin
sig_sin_spec = np.abs(np.fft.fft(sig_sin))
#plt.plot(sig_sin_spec)
sin_freqs = np.fft.fftfreq(sig_sin.shape[0], dt)


plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin)

plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-10, 10)
plt.plot(sin_freqs, sig_sin_spec)

plt.show()

# Test signal sawtooth
sig_saw = test_sig_ampl * signal.sawtooth(t * 2 * np.pi * test_sig_freq)

# Spectre test sawtooth
sig_saw_spec = np.abs(np.fft.fft(sig_saw))
saw_freqs = np.fft.fftfreq(sig_saw.shape[0], dt)


plt.subplot(1, 2, 1)
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlabel('Частота, Гц')
plt.xlim(0, 1)
plt.grid()
plt.plot(t, sig_saw)

plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-55, 55)
plt.plot(saw_freqs, sig_saw_spec)

plt.show()

# Test signal square
sig_square = test_sig_ampl * signal.square(t * 2 * np.pi * test_sig_freq)

# Spectre test square
sig_square_spec = np.abs(np.fft.fft(sig_square))
square_freqs = np.fft.fftfreq(sig_square.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlabel('Частота, Гц')
plt.xlim(0, 1)
plt.plot(t, sig_square)

plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel('Частота, Гц')
plt.plot(square_freqs, sig_square_spec)

plt.show()


# Синус через реле
sig_sin_relay = np.sign(sig_sin)
# Sin, relay and spectre
sig_sin_relay_spectre = np.abs(np.fft.fft(sig_sin_relay))
sig_sin_relay_freqs = np.fft.fftfreq(sig_sin_relay.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_relay)
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(sig_sin_relay_freqs, sig_sin_relay_spectre)

plt.show()

# Меандр через реле
sig_square_relay = np.sign(sig_square)
# spectre square after relay
sig_square_relay_spec = np.abs(np.fft.fft(sig_square_relay))
sig_square_relay_freqs = np.fft.fftfreq(sig_square_relay.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_relay)
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(sig_square_relay_freqs, sig_square_relay_spec)

plt.show()

# Пила через реле
sig_saw_relay = np.sign(sig_saw)
# Sawtooth relay spectre
sig_saw_relay_spec = np.abs(np.fft.fft(sig_saw_relay))
sig_saw_relay_freqs = np.fft.fftfreq(sig_saw_relay.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_relay)
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(sig_square_relay_freqs, sig_saw_relay_spec)

plt.show()
# function non linear element - dead zone
def dead_zone_scalar(x, wight = 0.5):
    if np.abs(x) < wight:
        return 0
    elif x > 0:
        return x - wight
    else:
        return x + wight

dead_zone = np.vectorize(dead_zone_scalar, otypes = [np.float], excluded = ['width'])


# Синус через мертвую зону
sig_sin_dz = dead_zone(sig_sin, non_lin_param_1)
# Sin dead zone spectre
sig_sin_dz_spec = np.abs(np.fft.fft(sig_sin_dz))
sig_sin_dz_freqs = np.fft.fftfreq(sig_sin_dz.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_dz)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_dz_freqs, sig_sin_dz_spec)

plt.show()


# Меандр через мертвую зону
sig_square_dz = dead_zone(sig_square, non_lin_param_1)
# Square after dead zone and spectre
sig_square_dz_spec = np.abs(np.fft.fft(sig_sin_dz))
sig_square_dz_freqs = np.fft.fftfreq(sig_sin_dz.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_dz)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_dz_freqs, sig_square_dz_spec)

plt.show()


# Пила через мертвую зону
sig_saw_dz = dead_zone(sig_saw, non_lin_param_1)
# Sawtooth after dead zone and spectre
sig_saw_dz_spec = np.abs(np.fft.fft(sig_saw_dz))
sig_saw_dz_freqs = np.fft.fftfreq(sig_saw_dz.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_dz)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_dz_freqs,sig_saw_dz_spec)

plt.show()


def saturation_scalar(x, hight = 0.5):
    if np.abs(x) < hight:
        return x
    elif x < hight:
        return -hight
    else:
        return hight

saturation = np.vectorize(saturation_scalar, otypes = [np.float], excluded = ['hight'])


# синус через насыщение
sig_sin_sat = saturation(sig_sin, non_lin_param_1)
# Sin after saturation spectre
sig_sin_sat_spec = np.abs(np.fft.fft(sig_sin_sat))
sig_sin_sat_freqs = np.fft.fftfreq(sig_sin_sat.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_sat)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_sat_freqs, sig_sin_sat_spec)

plt.show()

# меандр через насыщение
sig_square_sat = saturation(sig_square, non_lin_param_1)
#
sig_square_sat_spec = np.abs(np.fft.fft(sig_square_sat))
sig_square_sat_freqs = np.fft.fftfreq(sig_square_sat.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_sat)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_sat_freqs, sig_square_sat_spec)

plt.show()


# пила через насыщение
sig_saw_sat = saturation(sig_saw, non_lin_param_1)
# Saw after saturation spectre
sig_saw_sat_spec = np.abs(np.fft.fft(sig_saw_sat))
sig_saw_sat_freqs = np.fft.fftfreq(sig_saw_sat.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_sat)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_sat_freqs, sig_saw_sat_spec)

plt.show()


# Signals after non linear element and after lfilter
# Create linear element
#k = lin_param_k
#T = lin_param_T
B = [ lin_param_k / (1 + lin_param_T / dt) ]
A = [1, -1 / (1 + dt / lin_param_T)]

# Sin after relay and after lfilter
sig_sin_relay_lb = signal.lfilter(B, A, sig_sin_relay)
# Spectre
sig_sin_relay_lb_spec = np.abs(np.fft.fft(sig_sin_relay_lb))
sig_sin_relay_lb_freqs = np.fft.fftfreq(sig_sin_relay_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_relay, t, sig_sin_relay_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_relay_lb_freqs, sig_sin_relay_lb_spec)

plt.show()

# Square after realy and after lfilter
sig_square_relay_lb = signal.lfilter(B, A, sig_square_relay)
# Spectre
sig_square_relay_lb_spec = np.abs(np.fft.fft(sig_square_relay_lb))
sig_square_relay_lb_freqs = np.fft.fftfreq(sig_square_relay_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_relay, t, sig_square_relay_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_relay_lb_freqs, sig_square_relay_lb_spec)

plt.show()

# Sawtooth after realy and after lfilter
sig_saw_relay_lb = signal.lfilter(B, A, sig_saw_relay)
# Spectre
sig_saw_relay_lb_spec = np.abs(np.fft.fft(sig_saw_relay_lb))
sig_saw_relay_lb_freqs = np.fft.fftfreq(sig_saw_relay_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_relay, t, sig_saw_relay_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_relay_lb_freqs, sig_saw_relay_lb_spec)

plt.show()

# Sin after dead zone and after lfilter
sig_sin_dz_lb = signal.lfilter(B, A, sig_sin_dz)
# Spectre
sig_sin_dz_lb_spec = np.abs(np.fft.fft(sig_sin_dz_lb))
sig_sin_dz_lb_freqs = np.fft.fftfreq(sig_sin_dz_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_dz, t, sig_sin_dz_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_dz_lb_freqs, sig_sin_dz_lb_spec)

plt.show()

# Square after dead zone and after lfilter
sig_square_dz_lb = signal.lfilter(B, A, sig_square_dz)
# Spectre
sig_square_dz_lb_spec = np.abs(np.fft.fft(sig_square_dz_lb))
sig_square_dz_lb_freqs = np.fft.fftfreq(sig_square_dz_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_dz, t, sig_square_dz_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_dz_lb_freqs, sig_square_dz_lb_spec)

plt.show()

# Sawtooth after dead zone and after lfilter
sig_saw_dz_lb = signal.lfilter(B, A, sig_saw_dz)
# Spectre
sig_saw_dz_lb_spec = np.abs(np.fft.fft(sig_saw_dz_lb))
sig_saw_dz_lb_freqs = np.fft.fftfreq(sig_saw_dz_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_dz, t, sig_saw_dz_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_dz_lb_freqs, sig_saw_dz_lb_spec)

plt.show()

# Sin after saturation and after lfilter
sig_sin_sat_lb = signal.lfilter(B, A, sig_sin_sat)
# Spectre
sig_sin_sat_lb_spec = np.abs(np.fft.fft(sig_sin_sat_lb))
sig_sin_sat_lb_freqs = np.fft.fftfreq(sig_sin_sat_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_sat, t, sig_sin_sat_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_sat_lb_freqs, sig_sin_sat_lb_spec)

plt.show()

# Square after saturation and after lfilter
sig_square_sat_lb = signal.lfilter(B, A, sig_square_sat)
# Spectre
sig_square_sat_lb_spec = np.abs(np.fft.fft(sig_square_sat_lb))
sig_square_sat_lb_freqs = np.fft.fftfreq(sig_square_sat_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_sat, t, sig_square_sat_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_sat_lb_freqs, sig_square_sat_lb_spec)

plt.show()

# Sawtooth after saturation and after lfilter
sig_saw_sat_lb = signal.lfilter(B, A, sig_saw_sat)
# Spectre
sig_saw_sat_lb_spec = np.abs(np.fft.fft(sig_saw_relay_lb))
sig_saw_sat_lb_freqs = np.fft.fftfreq(sig_saw_sat_lb.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_sat, t, sig_saw_sat_lb)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_sat_lb_freqs, sig_saw_sat_lb_spec)

plt.show()

# Signal after linear element and after non linear

# Sin after lfilter and after relay
sig_sin_ln = signal.lfilter(B, A, sig_sin)
sig_sin_ln_relay = np.sign(sig_sin_ln)
# Spectre
sig_sin_ln_relay_spec = np.abs(np.fft.fft(sig_sin_ln_relay))
sig_sin_ln_relay_freqs = np.fft.fftfreq(sig_sin_ln_relay.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_ln, t, sig_sin_ln_relay)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_ln_relay_freqs, sig_sin_ln_relay_spec)

plt.show()

# Square after lfilter and after relay
sig_square_ln = signal.lfilter(B, A, sig_square)
sig_square_ln_relay = np.sign(sig_square_ln)
# Spectre
sig_square_ln_relay_spec = np.abs(np.fft.fft(sig_square_ln_relay))
sig_square_ln_relay_freqs = np.fft.fftfreq(sig_square_ln_relay.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_ln, t, sig_square_ln_relay)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_ln_relay_freqs, sig_sin_ln_relay_spec)

plt.show()

# Swatooth after lfilter and after realy
sig_saw_ln = signal.lfilter(B, A, sig_saw)
sig_saw_ln_relay = np.sign(sig_saw_ln)
# Spectre
sig_saw_ln_relay_spec = np.abs(np.fft.fft(sig_saw_ln_relay))
sig_saw_ln_relay_freqs = np.fft.fftfreq(sig_saw_ln_relay.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_ln, t, sig_saw_ln_relay)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_ln_relay_freqs, sig_saw_ln_relay_spec)

plt.show()

# Sin after lfilter and after dead zone
sig_sin_ln = signal.lfilter(B, A, sig_sin)
sig_sin_ln_dz = dead_zone(sig_sin_ln, non_lin_param_1)
# Spectre
sig_sin_ln_dz_spec = np.abs(np.fft.fft(sig_sin_ln_dz))
sig_sin_ln_dz_freqs = np.fft.fftfreq(sig_sin_ln_dz.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_ln, t, sig_sin_ln_dz)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_ln_dz_freqs, sig_sin_ln_dz_spec)

plt.show()

# Square after lfilter and after dead zone
sig_square_ln = signal.lfilter(B, A, sig_square)
sig_square_ln_dz = dead_zone(sig_square_ln, non_lin_param_1)
# Spectre
sig_square_ln_dz_spec = np.abs(np.fft.fft(sig_square_ln_dz))
sig_square_ln_dz_freqs = np.fft.fftfreq(sig_square_ln_dz.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_ln, t, sig_square_ln_dz)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_ln_dz_freqs, sig_square_ln_dz_spec)

plt.show()

# Swatooth after lfilter and after dead zone
sig_saw_ln = signal.lfilter(B, A, sig_saw)
sig_saw_ln_dz = dead_zone(sig_saw_ln, non_lin_param_1)
# Spectre
sig_saw_ln_dz_spec = np.abs(np.fft.fft(sig_saw_ln_dz))
sig_saw_ln_dz_freqs = np.fft.fftfreq(sig_saw_ln_dz.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_ln, t, sig_saw_ln_dz)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_ln_dz_freqs, sig_saw_ln_dz_spec)

plt.show()

# Sin after lfilter and after saturation
sig_sin_ln = signal.lfilter(B, A, sig_sin)
sig_sin_ln_sat = saturation(sig_sin_ln, non_lin_param_1)
# Spectre
sig_sin_ln_sat_spec = np.abs(np.fft.fft(sig_sin_ln_sat))
sig_sin_ln_sat_freqs = np.fft.fftfreq(sig_sin_ln_sat.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_ln, t, sig_sin_ln_sat)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_sin_ln_sat_freqs, sig_sin_ln_sat_spec)

plt.show()

# Square after lfilter and after relay
sig_square_ln = signal.lfilter(B, A, sig_square)
sig_square_ln_sat = saturation(sig_square_ln, non_lin_param_1)
# Spectre
sig_square_ln_sat_spec = np.abs(np.fft.fft(sig_square_ln_sat))
sig_square_ln_sat_freqs = np.fft.fftfreq(sig_square_ln_sat.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_square, t, sig_square_ln, t, sig_square_ln_sat)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_square_ln_sat_freqs, sig_square_ln_sat_spec)

plt.show()

# Swatooth after lfilter and after realy
sig_saw_ln = signal.lfilter(B, A, sig_saw)
sig_saw_ln_sat = saturation(sig_saw_ln, non_lin_param_1)
# Spectre
sig_saw_ln_sat_spec = np.abs(np.fft.fft(sig_saw_ln_sat))
sig_saw_ln_sat_freqs = np.fft.fftfreq(sig_saw_ln_sat.shape[0], dt)

plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-1.5, 1.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_ln, t, sig_saw_ln_sat)
plt.subplot(1, 2, 2)
plt.grid()
plt.xlabel("Частота, Гц")
plt.plot(sig_saw_ln_sat_freqs, sig_saw_ln_sat_spec)

plt.show()
