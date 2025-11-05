# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.colors as mcolors
import io
from datetime import datetime
import warnings

# Импорты для Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Игнорировать предупреждения
warnings.filterwarnings("ignore")

@dataclass
class Segment:
    start: int
    end: int
    data: np.ndarray
    mean: float
    std: float
    accepted: bool

@dataclass
class Shelf:
    start: int
    end: int
    data: np.ndarray
    mean: float
    std: float
    length: int
    segments_combined: int

class ShelfAnalyzer:
    def __init__(self):
        self.steps = {
            'window_size': 10,
            'std_threshold': 0.0001,
            'max_gap': 1,
            'min_shelf_length': 10,
            'sigma': 0.5
        }
        
    def load_and_preview_data(self, file_content):
        """Загрузка и предпросмотр данных"""
        try:
            data = pd.read_csv(file_content, header=None, encoding="ISO-8859-1", 
                              delimiter=';', decimal='.')
            return data
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            return None
    
    def get_numeric_columns(self, data):
        """Получение списка числовых колонок"""
        numeric_cols = []
        for col in range(data.shape[1]):
            try:
                numeric_data = pd.to_numeric(data.iloc[1:, col], errors='coerce')
                if not numeric_data.isna().all():
                    numeric_cols.append(col)
            except:
                continue
        return numeric_cols
    
    def filter_data(self, data, time_col, data_col, sigma):
        """Фильтрация данных с выбранными колонками"""
        time_str = data.iloc[1:, time_col].values
        time = pd.to_datetime(time_str, errors='coerce')
        
        wl = pd.to_numeric(data.iloc[1:, data_col], errors='coerce').values
        
        mask = ~np.isnan(wl) & ~pd.isna(time)
        wl = wl[mask]
        time = time[mask]
        
        if len(wl) == 0:
            raise ValueError("Нет валидных числовых данных в выбранной колонке")
        
        wl_filtered = gaussian_filter1d(wl, sigma=sigma)
        
        return wl, wl_filtered, time
    
    def analyze_segments(self, data, window_size, std_threshold):
        """Оптимизированный анализ сегментов на стабильность"""
        segments = []
        n_segments = len(data) // window_size
        
        for i in range(n_segments):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            segment_data = data[start_idx:end_idx]
            segment_std = np.std(segment_data, ddof=1)
            segment_mean = np.mean(segment_data)
            accepted = segment_std < std_threshold
            
            segments.append(Segment(
                start=start_idx,
                end=end_idx,
                data=segment_data,
                mean=segment_mean,
                std=segment_std,
                accepted=accepted
            ))
        
        return segments
    
    def merge_continuous_segments(self, segments, data, max_gap, min_shelf_length):
        """Объединяет непрерывные принятые сегменты в полки"""
        accepted_segments = [s for s in segments if s.accepted]
        
        if not accepted_segments:
            return []
        
        accepted_segments.sort(key=lambda x: x.start)
        shelves = []
        current_shelf = [accepted_segments[0]]
        
        for i in range(1, len(accepted_segments)):
            current_segment = accepted_segments[i]
            last_segment = current_shelf[-1]
            
            if current_segment.start <= last_segment.end + max_gap:
                current_shelf.append(current_segment)
            else:
                if len(current_shelf) > 0:
                    shelf = self.create_shelf_from_segments(current_shelf, data)
                    if shelf.length >= min_shelf_length:
                        shelves.append(shelf)
                current_shelf = [current_segment]
        
        if len(current_shelf) > 0:
            shelf = self.create_shelf_from_segments(current_shelf, data)
            if shelf.length >= min_shelf_length:
                shelves.append(shelf)
        
        return shelves
    
    def create_shelf_from_segments(self, segments, data):
        """Создает полку из объединенных сегментов"""
        start = segments[0].start
        end = segments[-1].end
        shelf_data = data[start:end]
        
        return Shelf(
            start=start,
            end=end,
            data=shelf_data,
            mean=np.mean(shelf_data),
            std=np.std(shelf_data, ddof=1),
            length=end - start,
            segments_combined=len(segments)
        )

def create_interactive_plotly_figure(time_seconds, wl_raw, wl_filtered, segments, shelves, title, chart_type):
    """Создает интерактивные высококачественные графики с Plotly"""
    
    # Цветовая схема - 4 цвета, которые будут повторяться
    base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Красный, бирюзовый, синий, зеленый
    
    fig = go.Figure()
    
    if chart_type == "overview":
        # Общий вид
        if len(wl_raw) > 1000:
            fig.add_trace(go.Scatter(
                x=time_seconds, y=wl_raw,
                mode='markers',
                marker=dict(size=2, color='#6B7280', opacity=0.4),
                name='Исходные данные',
                hovertemplate='<b>Время:</b> %{x:.2f} с<br><b>Значение:</b> %{y:.6f}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=time_seconds, y=wl_raw,
                mode='lines',
                line=dict(color='#6B7280', width=1),
                name='Исходные данные',
                hovertemplate='<b>Время:</b> %{x:.2f} с<br><b>Значение:</b> %{y:.6f}<extra></extra>'
            ))
        
        fig.add_trace(go.Scatter(
            x=time_seconds, y=wl_filtered,
            mode='lines',
            line=dict(color='#EF4444', width=2),
            name='Фильтрованные данные',
            hovertemplate='<b>Время:</b> %{x:.2f} с<br><b>Значение:</b> %{y:.6f}<extra></extra>'
        ))
        
    elif chart_type == "segments":
        # Сегменты
        fig.add_trace(go.Scatter(
            x=time_seconds, y=wl_filtered,
            mode='lines',
            line=dict(color='#6B7280', width=1, dash='dot'),
            showlegend=False,
            name='Все данные',
            opacity=0.3,
            hovertemplate='<b>Время:</b> %{x:.2f} с<br><b>Значение:</b> %{y:.6f}<extra></extra>'
        ))
        
        accepted_segments = [s for s in segments if s.accepted]
        
        for i, segment in enumerate(accepted_segments):
            segment_time = time_seconds[segment.start:segment.end]
            color = base_colors[i % len(base_colors)]
            
            fig.add_trace(go.Scatter(
                x=segment_time, y=segment.data,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False,
                name=f'Сегмент {i+1}',
                hovertemplate='<b>Сегмент %{customdata}</b><br>Время: %{x:.2f} с<br>Значение: %{y:.6f}<extra></extra>',
                customdata=[i+1] * len(segment_time)
            ))
            
    elif chart_type == "shelves":
        # Полки
        fig.add_trace(go.Scatter(
            x=time_seconds, y=wl_filtered,
            mode='lines',
            line=dict(color='#6B7280', width=1, dash='dot'),
            name='Все данные',
            opacity=0.3,
            hovertemplate='<b>Время:</b> %{x:.2f} с<br><b>Значение:</b> %{y:.6f}<extra></extra>'
        ))
        
        for i, shelf in enumerate(shelves):
            shelf_time = time_seconds[shelf.start:shelf.end]
            color = base_colors[i % len(base_colors)]
            
            # Основная линия полки
            fig.add_trace(go.Scatter(
                x=shelf_time, y=shelf.data,
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False,
                name=f'Полка {i+1}',
                hovertemplate='<b>Полка %{customdata}</b><br>Время: %{x:.2f} с<br>Значение: %{y:.6f}<extra></extra>',
                customdata=[i+1] * len(shelf_time)
            ))
            
            # Вертикальные линии границ полки
            fig.add_trace(go.Scatter(
                x=[shelf_time[0], shelf_time[0]],
                y=[min(shelf.data), max(shelf.data)],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                hovertemplate='<b>Начало полки %{customdata}</b><br>Время: %{x:.2f} с<extra></extra>',
                customdata=[i+1]
            ))
            
            fig.add_trace(go.Scatter(
                x=[shelf_time[-1], shelf_time[-1]],
                y=[min(shelf.data), max(shelf.data)],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hovertemplate='<b>Конец полки %{customdata}</b><br>Время: %{x:.2f} с<extra></extra>',
                customdata=[i+1]
            ))
    
    # Настройка layout для высокого качества
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='#1F2937', family="Arial, sans-serif")
        ),
        xaxis_title=dict(
            text='Время (секунды)',
            font=dict(size=12, color='#1F2937')
        ),
        yaxis_title=dict(
            text='Значение',
            font=dict(size=12, color='#1F2937')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        height=500,
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial",
            bordercolor="rgba(0,0,0,0.1)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=chart_type != "shelves"
    )
    
    # Настройка осей для лучшего качества
    # fig.update_xaxis(
    #     gridcolor='rgba(0,0,0,0.1)',
    #     zerolinecolor='rgba(0,0,0,0.2)',
    #     showgrid=True,
    #     mirror=True,
    #     ticks='outside',
    #     showline=True,
    #     linecolor='rgba(0,0,0,0.2)'
    # )
    
    # fig.update_yaxis(
    #     gridcolor='rgba(0,0,0,0.1)',
    #     zerolinecolor='rgba(0,0,0,0.2)',
    #     showgrid=True,
    #     mirror=True,
    #     ticks='outside',
    #     showline=True,
    #     linecolor='rgba(0,0,0,0.2)'
    # )
    
    return fig

def export_graph_data(time_seconds, wl_raw, wl_filtered, segments, shelves, chart_type):
    """Экспортирует данные графика в CSV"""
    
    if chart_type == "overview":
        # Данные для общего вида
        df = pd.DataFrame({
            'time_seconds': time_seconds,
            'raw_data': wl_raw,
            'filtered_data': wl_filtered
        })
        
    elif chart_type == "segments":
        # Данные для сегментов
        df = pd.DataFrame({
            'time_seconds': time_seconds,
            'filtered_data': wl_filtered
        })
        
        # Добавляем данные по сегментам
        accepted_segments = [s for s in segments if s.accepted]
        for i, segment in enumerate(accepted_segments):
            segment_time = time_seconds[segment.start:segment.end]
            segment_df = pd.DataFrame({
                'time_seconds': segment_time,
                f'segment_{i+1}': segment.data
            })
            df = pd.merge(df, segment_df, on='time_seconds', how='left')
        
    elif chart_type == "shelves":
        # Данные для полок
        df = pd.DataFrame({
            'time_seconds': time_seconds,
            'filtered_data': wl_filtered
        })
        
        # Добавляем данные по полкам
        for i, shelf in enumerate(shelves):
            shelf_time = time_seconds[shelf.start:shelf.end]
            shelf_df = pd.DataFrame({
                'time_seconds': shelf_time,
                f'shelf_{i+1}': shelf.data,
                f'shelf_{i+1}_start': [shelf_time[0]] * len(shelf_time),
                f'shelf_{i+1}_end': [shelf_time[-1]] * len(shelf_time)
            })
            df = pd.merge(df, shelf_df, on='time_seconds', how='left')
    
    return df

def main():
    # Инициализация page config с двумя боковыми панелями
    try:
        st.set_page_config(
            page_title="Shelf Finder",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass
    
    # Компактный CSS
    st.markdown("""
    <style>
    .compact-header {
        font-size: 1.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .compact-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 0.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-number {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.7rem;
        opacity: 0.9;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
    }
    .export-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Компактный заголовок
    st.markdown('<h1 class="compact-header">Shelf Finder</h1>', unsafe_allow_html=True)
    
    # Инициализация анализатора
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ShelfAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Левая боковая панель - параметры
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
            <h3 style="margin:0; font-size: 1.2rem;">⚙️ Управление</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Загрузка файла
        uploaded_file = st.file_uploader("Загрузить CSV", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"{uploaded_file.name}")
            
            # Предпросмотр данных и выбор колонок
            data_preview = analyzer.load_and_preview_data(uploaded_file)
            
            if data_preview is not None:
                st.markdown("**Колонки**")
                
                # Выбор колонки времени
                time_col = st.selectbox(
                    "Время",
                    options=range(data_preview.shape[1]),
                    format_func=lambda x: f"Колонка {x}",
                    help="Колонка с временными метками"
                )
                
                # Выбор колонки данных
                numeric_cols = analyzer.get_numeric_columns(data_preview)
                if not numeric_cols:
                    st.error("Нет числовых колонок")
                else:
                    data_col = st.selectbox(
                        "Данные",
                        options=numeric_cols,
                        format_func=lambda x: f"Колонка {x}",
                        help="Колонка с анализируемыми данными"
                    )
        
        st.markdown("**Параметры**")
        
        # Компактные слайдеры
        sigma = st.slider("σ Фильтрация", 0.1, 20.0, 5.0, 0.5)
        window_size = st.slider("Размер окна", 10, 500, 50, 10)
        std_threshold = st.slider("Порог STD", 0.00001, 0.01, 0.0004, 0.0001, format="%.5f")
        max_gap = st.slider("Макс. разрыв", 1, 20, 1, 1)
        min_shelf_length = st.slider("Мин. длина", 10, 1000, 50, 10)
        
        auto_update = st.checkbox("Автообновление", value=True)
        analyze_clicked = st.button("Запустить анализ", use_container_width=True) or (auto_update and uploaded_file is not None)
    
    # Правая боковая панель - результаты
    right_sidebar = st.sidebar
    with right_sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                    padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
            <h3 style="margin:0; font-size: 1.2rem;">📊 Результаты</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None and analyze_clicked and 'data_col' in locals():
            try:
                # Загрузка и обработка данных
                wl_raw, wl_filtered, time = analyzer.filter_data(
                    data_preview, time_col, data_col, sigma
                )
                
                # Анализ сегментов
                segments = analyzer.analyze_segments(
                    wl_filtered, window_size, std_threshold
                )
                
                # Объединение сегментов в полки
                shelves = analyzer.merge_continuous_segments(
                    segments, wl_filtered, max_gap, min_shelf_length
                )
                
                accepted = [s for s in segments if s.accepted]
                
                # Статистика полок
                st.markdown("**Статистика полок**")
                if shelves:
                    # Таблица полок для отображения
                    shelf_data = []
                    for i, shelf in enumerate(shelves, 1):
                        start_time = time[shelf.start]
                        end_time = time[shelf.end - 1]
                        
                        shelf_data.append({
                            '№': i,
                            'Начало': start_time.strftime('%H:%M:%S'),
                            'Конец': end_time.strftime('%H:%M:%S'),
                            'Длит.(с)': f"{(end_time - start_time).total_seconds():.1f}",
                            'Точки': shelf.length,
                            'Сегменты': shelf.segments_combined,
                            'Среднее': f"{shelf.mean:.6f}",
                            'STD': f"{shelf.std:.6f}",
                            'Отн.STD %': f"{(shelf.std/shelf.mean*100):.4f}"
                        })
                    
                    df_shelves = pd.DataFrame(shelf_data)
                    
                    # Таблица с возможностью экспорта CSV с разделителем ;
                    st.dataframe(df_shelves, use_container_width=True, height=300)
                    
                    # Создаем CSV с разделителем ; для экспорта через таблицу
                    csv_output = io.StringIO()
                    df_shelves.to_csv(csv_output, sep=';', index=False)
                    csv_data = csv_output.getvalue()
                    
                    st.download_button(
                        label="📥 Экспорт таблицы (CSV)",
                        data=csv_data,
                        file_name=f"shelves_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Экспорт результатов в TXT
                if shelves:
                    # Создаем текстовый файл с результатами (как раньше, без ;)
                    output = io.StringIO()
                    output.write("Shelf Finder - Результаты анализа\n")
                    output.write("=" * 50 + "\n\n")
                    output.write(f"Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    output.write(f"Файл: {uploaded_file.name}\n")
                    output.write(f"Колонка времени: {time_col}\n")
                    output.write(f"Колонка данных: {data_col}\n\n")
                    
                    output.write("ПАРАМЕТРЫ АНАЛИЗА:\n")
                    output.write(f"  Фильтрация (σ): {sigma}\n")
                    output.write(f"  Размер окна: {window_size}\n")
                    output.write(f"  Порог STD: {std_threshold}\n")
                    output.write(f"  Макс. разрыв: {max_gap}\n")
                    output.write(f"  Мин. длина полки: {min_shelf_length}\n\n")
                    
                    output.write("РЕЗУЛЬТАТЫ:\n")
                    output.write(f"  Всего сегментов: {len(segments)}\n")
                    output.write(f"  Принято сегментов: {len(accepted)}\n")
                    output.write(f"  Отклонено сегментов: {len(segments) - len(accepted)}\n")
                    output.write(f"  Образовано полок: {len(shelves)}\n\n")
                    
                    output.write("ПОЛКИ:\n")
                    output.write("-" * 80 + "\n")
                    # Красивое форматирование таблицы без разделителей
                    output.write(f"{'№':<3} {'Начало':<12} {'Конец':<12} {'Длит.(с)':<8} {'Точек':<6} {'Сегм.':<5} {'Среднее':<12} {'STD':<12} {'Отн.STD%':<10}\n")
                    output.write("-" * 80 + "\n")
                    
                    for i, shelf in enumerate(shelves, 1):
                        start_time = time[shelf.start]
                        end_time = time[shelf.end - 1]
                        duration = (end_time - start_time).total_seconds()
                        relative_std = (shelf.std/shelf.mean*100) if shelf.mean != 0 else 0
                        
                        output.write(f"{i:<3} {start_time.strftime('%H:%M:%S'):<12} {end_time.strftime('%H:%M:%S'):<12} ")
                        output.write(f"{duration:>7.1f} {shelf.length:>6} {shelf.segments_combined:>5} ")
                        output.write(f"{shelf.mean:>11.6f} {shelf.std:>11.6f} {relative_std:>9.4f}\n")
                    
                    txt_output = output.getvalue()
                    
                    st.download_button(
                        label="📄 Скачать полный отчет (TXT)",
                        data=txt_output,
                        file_name=f"shelf_finder_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
    
    # Основная область
    if uploaded_file is not None and analyze_clicked and 'data_col' in locals():
        try:
            with st.spinner("Анализ..."):
                # Загрузка и обработка данных
                wl_raw, wl_filtered, time = analyzer.filter_data(
                    data_preview, time_col, data_col, sigma
                )
                
                # Анализ сегментов
                segments = analyzer.analyze_segments(
                    wl_filtered, window_size, std_threshold
                )
                
                # Объединение сегментов в полки
                shelves = analyzer.merge_continuous_segments(
                    segments, wl_filtered, max_gap, min_shelf_length
                )
                
                # Визуализация
                time_seconds = (time - time[0]).total_seconds()
                
                # Компактная панель метрик
                accepted = [s for s in segments if s.accepted]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="compact-metric">
                        <div class="metric-number">{len(segments)}</div>
                        <div class="metric-label">Сегменты</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="compact-metric">
                        <div class="metric-number">{len(accepted)}</div>
                        <div class="metric-label">Принято</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="compact-metric">
                        <div class="metric-number">{len(shelves)}</div>
                        <div class="metric-label">Полки</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Инструкция по работе с графиками
                st.info("""
                **🎮 Управление графиками:** 
                - **🔍 Приближение**: Выделите область или используйте колесико мыши
                - **↔️ Перемещение**: Зажмите и перетаскивайте график  
                - **📏 Координаты**: Наведите курсор на точку данных
                - **🏠 Сброс**: Двойной клик или кнопка 'Autoscale'
                - **💾 Сохранение**: Нажмите на камеру в меню графика
                """)
                
                # Графики с Plotly
                # График 1: Общий вид
                with st.expander("📈 Общий вид", expanded=True):
                    fig1 = create_interactive_plotly_figure(
                        time_seconds, wl_raw, wl_filtered, segments, shelves,
                        f"Общий вид данных (σ={sigma})", "overview"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Экспорт данных для общего вида
                    with st.expander("💾 Экспорт данных графика", expanded=False):
                        st.markdown("**Данные графика 'Общий вид'**")
                        df_overview = export_graph_data(time_seconds, wl_raw, wl_filtered, segments, shelves, "overview")
                        
                        csv_overview = df_overview.to_csv(index=False, sep=';')
                        st.download_button(
                            label="📥 Скачать данные (CSV)",
                            data=csv_overview,
                            file_name=f"overview_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.dataframe(df_overview.head(10), use_container_width=True)
                        st.caption(f"Всего строк: {len(df_overview)}")
                
                # График 2: Сегменты
                with st.expander("🔍 Анализ сегментов", expanded=True):
                    fig2 = create_interactive_plotly_figure(
                        time_seconds, wl_raw, wl_filtered, segments, shelves,
                        f"Анализ сегментов (окно={window_size})", "segments"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Экспорт данных для сегментов
                    with st.expander("💾 Экспорт данных графика", expanded=False):
                        st.markdown("**Данные графика 'Анализ сегментов'**")
                        df_segments = export_graph_data(time_seconds, wl_raw, wl_filtered, segments, shelves, "segments")
                        
                        csv_segments = df_segments.to_csv(index=False, sep=';')
                        st.download_button(
                            label="📥 Скачать данные (CSV)",
                            data=csv_segments,
                            file_name=f"segments_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.dataframe(df_segments.head(10), use_container_width=True)
                        st.caption(f"Всего строк: {len(df_segments)}")
                
                # График 3: Полки
                with st.expander("🏆 Обнаруженные полки", expanded=True):
                    fig3 = create_interactive_plotly_figure(
                        time_seconds, wl_raw, wl_filtered, segments, shelves,
                        f"Обнаруженные полки (разрыв={max_gap})", "shelves"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Экспорт данных для полок
                    with st.expander("💾 Экспорт данных графика", expanded=False):
                        st.markdown("**Данные графика 'Обнаруженные полки'**")
                        df_shelves_data = export_graph_data(time_seconds, wl_raw, wl_filtered, segments, shelves, "shelves")
                        
                        csv_shelves = df_shelves_data.to_csv(index=False, sep=';')
                        st.download_button(
                            label="📥 Скачать данные (CSV)",
                            data=csv_shelves,
                            file_name=f"shelves_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.dataframe(df_shelves_data.head(10), use_container_width=True)
                        st.caption(f"Всего строк: {len(df_shelves_data)}")
                
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    else:
        # Компактная стартовая страница
        st.info("Загрузите CSV файл и настройте параметры")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Быстрый старт:
            1. Загрузите CSV с данными
            2. Выберите колонки времени и данных
            3. Настройте параметры анализа
            4. Запустите анализ
            
            ### Особенности:
            - Автоматическое определение колонок
            - Гибкая настройка параметров
            - **Высококачественные интерактивные графики**
            - **Экспорт данных графиков**
            - Экспорт результатов
            """)
        
        with col2:
            st.markdown("""
            ### Формат данных:
            CSV с разделителем ;
            - Временные метки
            - Числовые данные
            
            ### 🎮 Графики:
            - Приближение/отдаление
            - Перемещение
            - Определение координат
            - Сохранение изображений
            - **Экспорт данных**
            """)

if __name__ == "__main__":
    main()