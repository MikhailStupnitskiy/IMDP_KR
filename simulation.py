import streamlit as st
import simpy
import random
import pandas as pd
import altair as alt

# ===================== Конфигурация модели =====================
class Config:
    def __init__(self, n_channels=3, buffer_size=5, filter_strength=1.0, task_noise=0.1):
        self.N_CHANNELS = n_channels
        self.BUFFER_SIZE = buffer_size
        self.FILTER_STRENGTH = filter_strength
        self.TASK_NOISE_STD = task_noise
        self.MODES = {
            'Fast': 1.5,
            'Normal': 1.0,
            'Conservative': 0.7
        }

# ===================== Задача =====================
class Task:
    def __init__(self, task_id):
        self.id = task_id
        self.times = {}

# ===================== Канал / фильтр =====================
class PipelineChannel:
    def __init__(self, env, cfg, name):
        self.env = env
        self.cfg = cfg
        self.name = name
        self.buffer = simpy.Store(env)  # Без capacity, чтобы логировать реальные пики
        self.tasks_completed = 0
        self.history = []

        self.next_channel = None
        self.action = env.process(self.run())

    def choose_mode(self):
        q_len = len(self.buffer.items)
        if q_len > self.cfg.BUFFER_SIZE * 0.8:
            return 'Fast'
        elif q_len > self.cfg.BUFFER_SIZE * 0.4:
            return 'Normal'
        else:
            return 'Conservative'

    def run(self):
        while True:
            task = yield self.buffer.get()
            mode = self.choose_mode()

            base_time = random.uniform(0.5, 2.0)
            noise = random.gauss(0, self.cfg.TASK_NOISE_STD)
            duration = max(0.1, base_time + noise) \
                       / self.cfg.MODES[mode] \
                       / self.cfg.FILTER_STRENGTH

            yield self.env.timeout(duration)

            self.tasks_completed += 1
            task.times[self.name] = duration

            # Логируем реальную длину очереди
            real_queue_len = len(self.buffer.items)
            self.history.append({
                'Time': self.env.now,
                'Channel': self.name,
                'Queue Length': real_queue_len,
                'Mode': mode,
                'Task Duration': duration,
                'Task ID': task.id
            })

            if self.next_channel:
                yield self.next_channel.buffer.put(task)

# ===================== Процессор конвейера =====================
class PipelineProcessor:
    def __init__(self, env, cfg, n_tasks, task_interval=0.1):
        self.env = env
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.task_interval = task_interval

        self.channels = [
            PipelineChannel(env, cfg, f'Channel {i+1}')
            for i in range(cfg.N_CHANNELS)
        ]

        for i in range(len(self.channels) - 1):
            self.channels[i].next_channel = self.channels[i + 1]

        self.tasks = [Task(i) for i in range(n_tasks)]
        self.history = []

        env.process(self.run_pipeline())

    def run_pipeline(self):
        for task in self.tasks:
            yield self.channels[0].buffer.put(task)
            yield self.env.timeout(self.task_interval)  # Используем параметр task_interval

# ===================== Запуск симуляции =====================
def run_simulation(n_channels, buffer_size, filter_strength, task_noise, n_tasks, task_interval=0.1):
    env = simpy.Environment()
    cfg = Config(n_channels, buffer_size, filter_strength, task_noise)
    processor = PipelineProcessor(env, cfg, n_tasks, task_interval)

    env.run(until=n_tasks * 5)

    history = []
    for ch in processor.channels:
        history.extend(ch.history)

    df = pd.DataFrame(history)
    return df, processor

# ===================== Функция для серии экспериментов =====================
def run_experiments(experiment_type, param_range, fixed_params, n_tasks=30):
    results = []

    for val in param_range:
        n_channels = fixed_params.get('n_channels', 3)
        buffer_size = fixed_params.get('buffer_size', 5)
        filter_strength = fixed_params.get('filter_strength', 1.0)
        task_noise = fixed_params.get('task_noise', 0.1)
        task_interval = fixed_params.get('task_interval', 0.1)

        if experiment_type == 'A':  # варьируем количество каналов
            n_channels = val
        elif experiment_type == 'B':  # варьируем интенсивность потока задач
            task_interval = val
        elif experiment_type == 'C':  # варьируем силу фильтра
            filter_strength = val

        df, processor = run_simulation(n_channels, buffer_size, filter_strength, task_noise, n_tasks, task_interval)

        results.append({
            'param_value': val,
            'avg_task_duration': df['Task Duration'].mean(),
            'max_queue_length': df['Queue Length'].max(),
            'tasks_completed': len(processor.tasks)
        })

    return pd.DataFrame(results)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Pipeline Simulation", layout="wide", page_icon="⚙️")
st.title("⚙️ Симуляция конвейера задач (Channels & Filters)")
st.markdown("---")

with st.sidebar:
    st.header("Параметры модели")
    n_channels = st.slider("Количество каналов", 1, 5, 3)
    buffer_size = st.slider("Ёмкость буфера", 1, 10, 5)
    filter_strength = st.slider("Сила фильтра", 0.5, 2.0, 1.0, 0.1)
    task_noise = st.slider("Шум задач", 0.0, 1.0, 0.1, 0.05)
    n_tasks = st.slider("Количество задач", 10, 50, 30)

st.subheader("Сценарии экспериментов")
exp_mode = st.radio("Выберите режим:", ["Одиночная симуляция", "Серия экспериментов"])

if exp_mode == "Одиночная симуляция":
    if st.button("🚀 Запустить симуляцию"):
        with st.spinner("Выполняется симуляция..."):
            df, processor = run_simulation(n_channels, buffer_size, filter_strength, task_noise, n_tasks)

        st.subheader("📊 Основные показатели")
        c1, c2, c3 = st.columns(3)
        c1.metric("Обработано задач", n_tasks)
        c2.metric("Среднее время обработки", f"{df['Task Duration'].mean():.2f} сек")
        c3.metric("Макс длина очереди", int(df['Queue Length'].max()))

        st.subheader("📈 Динамика очередей по каналам")
        st.altair_chart(
            alt.Chart(df).mark_line().encode(
                x='Time',
                y='Queue Length',
                color='Channel',
                tooltip=['Time','Channel','Queue Length','Mode','Task Duration','Task ID']
            ).properties(height=300),
            use_container_width=True
        )

        st.subheader("🟢 Распределение времени обработки задач")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X('Task Duration', bin=alt.Bin(maxbins=20)),
                y='count()',
                color='Channel'
            ).properties(height=300),
            use_container_width=True
        )

        st.subheader("🔥 Прохождение задач по каналам (Gantt)")
        df['Time_start'] = df['Time'] - df['Task Duration']

        st.altair_chart(
            alt.Chart(df).mark_rect().encode(
                x='Time_start',
                x2='Time',
                y='Channel',
                color=alt.Color('Task Duration', scale=alt.Scale(scheme='inferno')),
                tooltip=['Task ID','Channel','Task Duration','Mode','Queue Length']
            ).properties(height=300),
            use_container_width=True
        )

else:  # Серия экспериментов
    st.write("В экспериментах варьируются ключевые параметры конвейера:")

    exp_type = st.selectbox("Выберите эксперимент:", ["A – количество каналов", "B – интенсивность потока задач", "C – сила фильтра"])

    if st.button("🚀 Запустить эксперимент"):
        if exp_type == "A – количество каналов":
            param_range = list(range(1, 6))
            fixed_params = {'buffer_size': buffer_size, 'filter_strength': filter_strength, 'task_noise': task_noise, 'task_interval': 0.1}
            df_exp = run_experiments('A', param_range, fixed_params, n_tasks)

        elif exp_type == "B – интенсивность потока задач":
            param_range = [0.05, 0.1, 0.2, 0.3, 0.5]
            fixed_params = {'n_channels': n_channels, 'buffer_size': buffer_size, 'filter_strength': filter_strength, 'task_noise': task_noise}
            df_exp = run_experiments('B', param_range, fixed_params, n_tasks)

        elif exp_type == "C – сила фильтра":
            param_range = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
            fixed_params = {'n_channels': n_channels, 'buffer_size': buffer_size, 'task_noise': task_noise, 'task_interval': 0.1}
            df_exp = run_experiments('C', param_range, fixed_params, n_tasks)

        # Построение графиков
        chart1 = alt.Chart(df_exp).mark_line(point=True).encode(
            x='param_value',
            y='avg_task_duration',
            tooltip=['param_value','avg_task_duration','max_queue_length','tasks_completed'],
            color=alt.value('blue')
        ).properties(title="Среднее время обработки задач")

        chart2 = alt.Chart(df_exp).mark_line(point=True).encode(
            x='param_value',
            y='max_queue_length',
            tooltip=['param_value','avg_task_duration','max_queue_length','tasks_completed'],
            color=alt.value('red')
        ).properties(title="Максимальная длина очереди")

        st.altair_chart(chart1 & chart2, use_container_width=True)
