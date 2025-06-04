import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# Создаем среду
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")  # Детерминированная версия

# Инициализация Q-таблицы
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Гиперпараметры
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 3000

rewards = []

# Цвета для визуализации
lake_colors = {
    'S': 'yellow',    # Старт
    'F': 'cyan',      # Лед
    'H': 'red',       # Лужа (опасность)
    'G': 'green',     # Цель
}


# Функция для отображения карты
def plot_lake(env, current_state=None):
    desc = env.desc.tolist() if hasattr(env, 'desc') else [
        ['S', 'F', 'F', 'F'],
        ['F', 'H', 'F', 'H'],
        ['F', 'F', 'F', 'H'],
        ['H', 'F', 'F', 'G']
    ]
    fig, ax = plt.subplots()
    for i in range(len(desc)):
        for j in range(len(desc[i])):
            cell = desc[i][j].decode('utf-8') if isinstance(desc[i][j], bytes) else desc[i][j]
            color = lake_colors.get(cell, 'white')
            if current_state is not None and i * len(desc) + j == current_state:
                color = 'blue'  # Текущая позиция агента
            ax.add_patch(plt.Rectangle((j, -i), 1, 1, color=color))
            ax.text(j + 0.5, -i + 0.5, cell, ha='center', va='center')
    ax.set_xlim(0, len(desc[0]))
    ax.set_ylim(-len(desc) + 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("FrozenLake-v1 (Текущее состояние: S{}".format(current_state))
    plt.show(block=False)
    plt.pause(2)
    plt.close()


# Обучение агента
for ep in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    # Визуализация первого эпизода
    if ep == 0:
        print("Визуализация первого эпизода:")
        plot_lake(env, state)

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)

        # Обновление Q-таблицы
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        # Визуализация первого эпизода
        if ep == 0:
            plot_lake(env, state)
            time.sleep(0.3)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

# Визуализация Q-таблицы (первые 10 строк)
print("\nQ-таблица (первые 10 строк):")
print(q_table[:10])

# Скользящее среднее
def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid') if len(x) >= w else x

# График наград
plt.figure(figsize=(10, 5))
plt.plot(moving_avg(rewards))
plt.title("Q-Learning: Награда по эпизодам (скользящее среднее, окно=50)")
plt.xlabel("Эпизоды")
plt.ylabel("Награда")
plt.grid()
plt.show()

# Визуализация одного эпизода после обучения
print("\nДемонстрация обученного агента:")
state = env.reset()[0]
done = False
while not done:
    action = np.argmax(q_table[state])
    next_state, _, done, _, _ = env.step(action)
    plot_lake(env, state)
    time.sleep(0.5)
    state = next_state
plot_lake(env, state)  # Финальное состояние