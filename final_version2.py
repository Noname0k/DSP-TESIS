import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import json
import librosa
import threading
import time
import queue
import csv
import traceback
#import spidev
#import RPi.GPIO as GPIO

from scipy import signal
from scipy.signal import find_peaks
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import deque
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

class ToolTip:
    """Clase para crear botones interactivos"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Arial", "10", "normal"))
        label.pack()

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class SignalProcessorApp:
    """Aplicación de procesamiento de señales para Raspberry Pi 4."""
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de Señales - Tesis UVG")
        
        self.screen_width = 800
        self.screen_height = 480
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.resizable(False, False)
        
        # Configurar el protocolo de cierre para manejar excepciones
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Large.TButton', font=('Arial', 12, 'bold'), padding=10)
        self.style.configure('Info.TButton', font=('Arial', 9), foreground='blue')
        
        # Variables para configuración global de ejes
        self.time_units = tk.StringVar(value="segundos")
        self.freq_units = tk.StringVar(value="Hz") 
        self.mag_units = tk.StringVar(value="Adimensional")
        self.phase_units = tk.StringVar(value="grados")
        
        # Variable para rastrear si hay una señal cargada
        self.signal_loaded = False
        
        self.input_signal, self.output_signal, self.current_filter = None, None, None
        self.fs = 44100
        self.saved_signals = {}
        
        self.processing_thread, self.realtime_stream, self.playback_stream = None, None, None
        self.stop_thread = threading.Event()
        self.realtime_plot_queue = queue.Queue()
        self.realtime_filter_zi = None
        self.is_paused, self.playback_position = False, 0
        self.current_playback_signal = None
        self.fig = None

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filters_dir = os.path.join(self.base_dir, "filtros_guardados")
        self.signals_dir = os.path.join(self.base_dir, "señales_guardadas")
        self.create_directories()
        
        # Gestionar la navegación
        self.managed_windows = {}
        self.navigation_stack = []  

        # Variables StringVars para Fs y Nyquist para actualización automática
        self.fs_display_var = tk.StringVar(value=f"Fs: {self.fs} Hz")
        self.nyquist_display_var = tk.StringVar(value=f"Nyquist: {self.fs/2:.0f} Hz")

        # Crear contenedor principal
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        #Optimizaciones de la Raspberry pi 4
        self._optimize_for_raspberry_pi()
        
        # Configurar estilos para mejores gráficas
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['grid.alpha'] = 0.3
        
        # Configurar límite máximo de muestras para buen rendimiento
        self.max_samples = 30 * 44100  # 30 segundos a 44.1 kHz
        
        # Hacer que los cambios en estas variables actualicen todas las gráficas
        def update_all_plots(*args):
            """Actualiza todas las gráficas cuando cambian las unidades globales"""
            # Actualizar gráfica de salida si existe
            if hasattr(self, 'current_frame') and hasattr(self.current_frame, 'update_plots'):
                self.current_frame.update_plots()
            
        self.time_units.trace('w', update_all_plots)
        self.freq_units.trace('w', update_all_plots)
        self.mag_units.trace('w', update_all_plots)
        
        self.setup_welcome_tab()
    
    def on_closing(self):
        """Maneja el cierre de la aplicación con verificación de cambios no guardados"""
        if self._has_unsaved_changes():
            response = messagebox.askyesnocancel(
                "Cambios no guardados",
                "Tienes señales cargadas que no han sido guardadas. ¿Deseas guardarlas antes de salir?\n\n"
                "Sí: Guardar y salir\n"
                "No: Descartar cambios y salir\n"
                "Cancelar: Permanecer en la aplicación"
            )
            
            if response is None:  # Cancelar
                return
            elif response:  # Sí - Guardar
                self._save_current_signal()
        
        # Continuar con el cierre normal
        try:
            # Detener todos los streams activos
            if hasattr(self, 'realtime_stream') and self.realtime_stream:
                try:
                    self.realtime_stream.stop()
                    self.realtime_stream.close()
                except Exception as e:
                    print(f"Error cerrando realtime stream: {e}")
            
            if hasattr(self, 'playback_stream') and self.playback_stream:
                try:
                    self.playback_stream.stop()
                    self.playback_stream.close()
                except Exception as e:
                    print(f"Error cerrando playback stream: {e}")
                    
            if hasattr(self, 'recording_stream') and self.recording_stream:
                try:
                    self.recording_stream.stop()
                    self.recording_stream.close()
                except Exception as e:
                    print(f"Error cerrando recording stream: {e}")
                    
            # Detener threads activos
            if hasattr(self, 'stop_thread'):
                self.stop_thread.set()
                
            # Cerrar todas las ventanas gestionadas
            for window, fig in self.managed_windows.items():
                try:
                    plt.close(fig)
                    window.destroy()
                except:
                    pass
                    
            self.root.destroy()
        except Exception as e:
            print(f"Error al cerrar la aplicación: {e}")
            try:
                self.root.destroy()
            except:
                os._exit(0)  # Salida forzada como último recurso
        
    def _optimize_for_raspberry_pi(self):
        """Optimiza la configuración para Raspberry Pi"""
        # Reducir la frecuencia de muestreo por defecto
        self.fs = 44100  # Valor por defecto suficiente para RPi
        
        # Configurar parámetros de audio optimizados
        sd.default.samplerate = self.fs
        sd.default.blocksize = 1024
        sd.default.latency = 'high'
        
        # Configurar matplotlib para bajo consumo
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 1.0
        plt.rcParams['agg.path.chunksize'] = 10000
        
        # Limitar el uso de memoria para gráficos
        self.max_plot_points = 3000    

    def compute_magnitude_spectrum(signal_data , fs , db_scale = False , detect_peaks = False , zero_padding_factor = 4 ) :
        """
        Calcula el espectro de magnitud unilateral de una señal real.
        
        Parámetros:
        - signal_data: array-like, señal de entrada (1D)
        - fs: float, frecuencia de muestreo en Hz
        - db_scale: bool, si True devuelve magnitud en dB, si False en magnitud lineal
        - zero_padding_factor: int, factor para zero-padding (por defecto 4)
        - detect_peaks: bool, si True detecta y muestra picos en el espectro
        
        Retorna:
        - frecuencias: array con frecuencias del espectro unilateral (0 a fs/2)
        - magnitud: array con magnitud o magnitud en dB del espectro unilateral
        """

        if signal_data is None:
            return None, None

        datos = np.asarray(signal_data, dtype=np.float64)
        datos = np.nan_to_num(datos)
        L = len(datos)

        if L < 2:
            return None, None

        # Zero-padding para mejorar resolución espectral
        N_fft = max(zero_padding_factor * L, 8192) # Si los 8192 puntos no es suficiente aumentar, ojo que se tarda más en hacer el proceso.
        # N_fft = max(2**int(np.ceil(np.log2(L))), 2048) #Hacer prueba con las dos y ver que funciona mejor

        # FFT con zero-padding
        Y = np.fft.fft(datos, n=N_fft)

        # Normalización por longitud original para mantener amplitud correcta
        P2 = np.abs(Y) / L

        # Espectro unilateral (mitad positiva)
        P1 = P2[:N_fft // 2 + 1]

        # Multiplicar por 2 las componentes intermedias para conservar energía total
        if N_fft % 2 == 0:
            if len(P1) > 2:
                P1[1:-1] *= 2
        else:
            if len(P1) > 1:
                P1[1:] *= 2

        # Vector de frecuencias
        frecuencias = np.linspace(0, fs / 2, N_fft // 2 + 1)

        # Detección de picos (opcional, comentar si no se quiere)
        """
        if detect_peaks:
            magnitud_suavizada = gaussian_filter1d(P1, sigma=1)
            picos, propiedades = find_peaks(
                magnitud_suavizada,
                height=np.max(P1) * 0.05 if np.max(P1) > 0 else 0.01,
                distance=10,
                prominence=0.01,
                width=2
            )
            if len(picos) > 0:
                print("COMPONENTES DETECTADAS:")
                for pico in picos:
                    freq = frecuencias[pico]
                    mag = P1[pico]
                    print(f"   {freq:.2f} Hz - Magnitud: {mag:.6f}")
        """
        # Escala en dB si se solicita
        if db_scale:
            P1 = 20 * np.log10(np.maximum(P1, 1e-12))

        # Debug técnico (opcional, comentar si no se quiere)
        """
        print("DEBUG Espectro:")
        print(f"   - Señal: min={np.min(datos):.6f}, max={np.max(datos):.6f}")
        print(f"   - FFT puntos: {N_fft}")
        print(f"   - Resolución: {frecuencias[1] - frecuencias[0]:.6f} Hz")
        print(f"   - Pico máximo: {np.max(P1):.6f}")
        print(f"   - Ventana: Rectangular (equivalente MATLAB)")
        """

        return frecuencias, P1       

    def limit_signal_length(self, signal, fs, max_duration=30):
        """
        Limita la longitud de la señal a un máximo de segundos
        para evitar problemas de rendimiento, manteniendo la definición
        """
        if signal is None:
            return None
            
        max_samples = int(max_duration * fs)
        if len(signal) > max_samples:
            # Limitar pero mantener la longitud original para cálculos
            return signal[:max_samples]
        return signal
        
    def apply_filter_optimized(self, signal_name, filter_name):
        """
        Versión optimizada con manejo de estabilidad
        """
        if not signal_name or not filter_name:
            messagebox.showwarning("Selección requerida", "Debes seleccionar una señal y un filtro.")
            return None
        
        try:
            # Cargar señal y filtro
            input_data, fs = self.saved_signals[signal_name]['data'], self.saved_signals[signal_name]['fs']
            filter_path = os.path.join(self.filters_dir, filter_name)
            
            with open(filter_path, 'r') as f:
                filter_info = json.load(f)
            
            # ACTUALIZAR Fs CON LA DE LA SEÑAL GUARDADA
            self.fs = fs
            self._update_fs_display()
            
            b, a = np.array(filter_info['b']), np.array(filter_info['a'])
            filter_coeffs = {'b': b, 'a': a}
            
            print(f"🔧 Aplicando filtro {filter_info.get('effect', '?')} {filter_info.get('type', '?')}")
            
            #   VERIFICAR ESTABILIDAD PARA IIR
            if filter_info.get('type') == 'IIR' and not self._es_filtro_estable(b, a):
                print(" Filtro IIR inestable, no se puede aplicar")
                return "filtro_inestable"
            
            #   APLICAR FILTRADO CON MÉTODO ESTABLE
            if filter_info.get('type') == 'IIR':
                # Para IIR usar sosfilt que es más estable
                print("Usando filtrado IIR estable (sosfilt)...")
                sos = signal.tf2sos(b, a)
                output_signal = signal.sosfilt(sos, input_data)
            else:
                # Para FIR usar lfilter normal
                print("Usando filtrado FIR...")
                output_signal = signal.lfilter(b, a, input_data)
            
            #   VERIFICAR QUE NO HAY NaN
            if np.any(np.isnan(output_signal)):
                print("Señal filtrada contiene NaN")
                # Intentar normalizar la señal de entrada
                print("Normalizando señal de entrada...")
                input_normalized = input_data / np.max(np.abs(input_data)) * 0.8
                if filter_info.get('type') == 'IIR':
                    output_signal = signal.sosfilt(sos, input_normalized)
                else:
                    output_signal = signal.lfilter(b, a, input_normalized)
                
                if np.any(np.isnan(output_signal)):
                    print("Persisten NaN después de normalización")
                    return "error_nan"
            
            #   VERIFICAR QUE LA SEÑAL NO ES CERO
            if np.max(np.abs(output_signal)) < 1e-10:
                print("Señal filtrada es prácticamente cero")
                return "senal_cero"
            
            print(f"Filtrado exitoso: entrada={np.max(np.abs(input_data)):.6f}, salida={np.max(np.abs(output_signal)):.6f}")
            
            # Calcular espectros
            frecuencias_x, magnitud_x = self.compute_magnitude_spectrum(input_data, fs)
            frecuencias_h, magnitud_h = self.compute_magnitude_spectrum(b, fs)
            frecuencias_y, magnitud_y = self.compute_magnitude_spectrum(output_signal, fs)
            
            resultado = {
                'y_n': output_signal,
                'frecuencias_x': frecuencias_x,
                'magnitud_x': magnitud_x,
                'frecuencias_h': frecuencias_h,
                'magnitud_h': magnitud_h,
                'frecuencias_y': frecuencias_y,
                'magnitud_y': magnitud_y
            }
            
            self.input_signal = input_data
            self.output_signal = output_signal
            
            return "éxito"
                
        except Exception as e:
            print(f"   Error en aplicación de filtro: {e}")
            import traceback
            traceback.print_exc()
            return str(e)
    
    def _es_filtro_estable(self, b, a):
        """
        Verifica si un filtro IIR es estable
        """
        try:
            # Los polos deben estar dentro del círculo unitario
            poles = np.roots(a)
            return all(np.abs(poles) < 1.0)
        except:
            return False
       
    # --------------------------------------------------------------------------
    # 1. MÉTODOS DE CONFIGURACIÓN Y UTILIDADES
    # --------------------------------------------------------------------------

    def create_directories(self):
        try:
            os.makedirs(self.filters_dir, exist_ok=True)
            os.makedirs(self.signals_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error de Directorio", f"No se pudieron crear los directorios: {e}")

    def _run_in_thread(self, target_func, callback_success=None, callback_finally=None):
        def threaded_task():
            try:
                result = target_func()
                if callback_success and self.root.winfo_exists():
                    self.root.after(0, lambda: callback_success(result))
            except Exception as e:
                if self.root.winfo_exists():
                    self.root.after(0, lambda: messagebox.showerror("Error en Proceso", str(e)))
                print(f"Error en hilo: {traceback.format_exc()}")
            finally:
                if callback_finally and self.root.winfo_exists():
                    self.root.after(0, callback_finally)
                if self.root.winfo_exists():
                    self.root.config(cursor="")
        
        # Verificar que la ventana principal todavía existe
        if not self.root.winfo_exists():
            return
            
        self.root.config(cursor="watch")
        thread = threading.Thread(target=threaded_task, daemon=True)
        thread.start()
        
    def _update_fs_display(self):
        """Actualiza las etiquetas de Fs y Nyquist en la UI."""
        self.fs_display_var.set(f"Fs: {self.fs} Hz")
        self.nyquist_display_var.set(f"Nyquist: {self.fs/2:.0f} Hz")
    
    def _create_info_button(self, parent, text, tooltip_text):
        """Crea un botón de información con tooltip"""
        btn = ttk.Button(parent, text=text, style='Info.TButton', width=3)
        ToolTip(btn, tooltip_text)
        return btn
    
    # --------------------------------------------------------------------------
    # 2. GESTIÓN DE VENTANAS Y NAVEGACIÓN (MODIFICADO)
    # --------------------------------------------------------------------------

    def _show_frame(self, frame_class, *args, **kwargs):
        """Muestra un nuevo frame, ocultando el actual y guardando la navegación"""
        # Ocultar frame actual si existe
        for widget in self.main_container.winfo_children():
            widget.pack_forget()
        
        # Guardar el frame actual en la pila de navegación
        if hasattr(self, 'current_frame'):
            self.navigation_stack.append(self.current_frame)
        
        # Crear y mostrar el nuevo frame
        try:
            self.current_frame = frame_class(self.main_container, self, *args, **kwargs)
            self.current_frame.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la pantalla: {str(e)}")
            print(f"Error al mostrar frame: {traceback.format_exc()}")
            self._go_back()

    def _go_back(self, check_unsaved=True):
        """Regresa al frame anterior en la navegación"""
        # Verificar cambios no guardados solo si se solicita y en frames específicos
        if check_unsaved and self._has_unsaved_changes():
            current_frame_type = type(self.current_frame).__name__
            input_frames = ['FileLoadOptionsFrame', 'SamplingOptionsFrame', 'RecordingOptionsFrame']
            
            if current_frame_type in input_frames:
                response = messagebox.askyesnocancel(
                    "Cambios no guardados",
                    "Tienes una señal cargada que no ha sido guardada. ¿Deseas guardarla antes de salir?\n\n"
                    "Sí: Guardar y volver\n"
                    "No: Descartar cambios y volver\n"
                    "Cancelar: Permanecer aquí"
                )
                
                if response is None:  # Cancelar
                    return
                elif response:  # Sí - Guardar
                    if self._save_current_signal():
                        # Si se guardó exitosamente, continuar
                        pass
                    else:
                        # Si canceló el guardado, permanecer aquí
                        return
                else:  # No - Descartar cambios
                    self._discard_unsaved_changes()
        
        # Continuar con la navegación normal
        if self.navigation_stack:
            self.current_frame.pack_forget()
            prev_frame = self.navigation_stack.pop()
            self.current_frame = prev_frame
            self.current_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self._show_frame(WelcomeFrame)
    
    def _has_unsaved_changes(self):
        """Verifica si hay señales cargadas que no han sido guardadas"""
        # Solo considerar cambios no guardados si hay señal de entrada cargada
        if self.input_signal is not None:
            # Verificar si la señal actual está guardada en saved_signals
            signal_is_saved = any(
                np.array_equal(self.input_signal, saved_signal['data'])
                for saved_signal in self.saved_signals.values()
            )
            return not signal_is_saved
        return False
    
    def _discard_unsaved_changes(self):
        """Descarta todas las señales no guardadas"""
        # Limpiar señales temporales
        self.input_signal = None
        self.output_signal = None
        self.signal_loaded = False
        
        # Limpiar el filtro actual si se desea (opcional)
        # self.current_filter = None
        
        # Limpiar recursos de audio
        self.cleanup_audio_resources()
        
        # Actualizar estado de botones de reproducción si existen
        if hasattr(self, 'playback_buttons'):
            for key in self.playback_buttons:
                self._update_playback_buttons_state(key, False)
        
        print("Señales no guardadas descartadas")

    def _show_window(self, title, content_func):
        """Muestra una ventana modal (solo para gráficas y ventanas especiales)"""
        try:
            window = tk.Toplevel(self.root)
            window.title(title)
            window.geometry(f"{self.screen_width}x{self.screen_height}")
            window.resizable(False, False)
            window.transient(self.root)  # Hacerla modal
            window.grab_set()  # Hacerla modal
            
            # Crear contenido
            content_func(window)
            
            # Añadir botón de volver
            ttk.Button(window, text="←", command=lambda: self._close_window(window)).pack(anchor="nw", padx=10, pady=10)
            
            # Centrar la ventana
            window.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (window.winfo_width() // 2)
            y = (self.root.winfo_screenheight() // 2) - (window.winfo_height() // 2)
            window.geometry(f"+{x}+{y}")
            
            # Configurar comportamiento al cerrar
            window.protocol("WM_DELETE_WINDOW", lambda: self._close_window(window))
            
            return window
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo crear la ventana: {str(e)}")
            print(f"Error al crear ventana: {traceback.format_exc()}")

    def _close_window(self, window):
        """Cierra una ventana modal"""
        try:
            window.grab_release()
            window.destroy()
        except:
            pass

    # --------------------------------------------------------------------------
    # 3. DEFINICIÓN DE FRAMES PARA CADA PANTALLA
    # --------------------------------------------------------------------------

    def setup_welcome_tab(self):
        """Configura la pestaña de bienvenida."""
        self._show_frame(WelcomeFrame)

    def show_mode_selection(self):
        """Muestra la ventana de selección de modo."""
        self._show_frame(ModeSelectionFrame)

    def show_stages_selection(self):
        """Muestra la selección de etapas (Entrada, Filtro, Salida)."""
        self._show_frame(StagesSelectionFrame)

    def show_input_options(self):
        """Muestra las opciones de entrada de señal."""
        self._show_frame(InputOptionsFrame)

    def show_sampling_options(self):
        """Muestra las opciones de muestreo simulado."""
        self._show_frame(SamplingOptionsFrame)

    def show_file_load_options(self):
        """Muestra las opciones de carga de archivo."""
        self._show_frame(FileLoadOptionsFrame)

    def show_recording_options(self):
        """Muestra las opciones de grabación."""
        self._show_frame(RecordingOptionsFrame)

    def show_filter_options(self):
        """Muestra las opciones de diseño de filtro."""
        self._show_frame(FilterOptionsFrame)

    def show_output_options(self):
        """Muestra las opciones de salida."""
        self._show_frame(OutputOptionsFrame)

    def show_realtime_processing(self):
        """Muestra el procesamiento en tiempo real."""
        self._show_frame(RealtimeProcessingFrame)

    # --------------------------------------------------------------------------
    # 4. MÉTODOS DE ENTRADA DE SEÑAL (Carga, Grabación, Muestreo)
    # --------------------------------------------------------------------------
    
    def _save_current_signal(self):
        """Guarda la señal actual en memoria con un nombre y retorna éxito"""
        if self.input_signal is None:
            messagebox.showwarning("Sin señal", "No hay señal para guardar.")
            return False
        
        name = simpledialog.askstring("Guardar Señal", "Nombre para la señal:", parent=self.root)
        if name:
            if name in self.saved_signals:
                if not messagebox.askyesno("Confirmar", f"Ya existe una señal llamada '{name}'. ¿Sobrescribir?"):
                    return False
            self.saved_signals[name] = {'data': self.input_signal.copy(), 'fs': self.fs}
            messagebox.showinfo("Éxito", f"Señal '{name}' guardada en memoria.")
            return True
        return False
    
    def _save_output_signal(self):
        """Guarda la señal de salida en memoria con un nombre"""
        if self.output_signal is None:
            messagebox.showwarning("Sin señal", "No hay señal de salida para guardar.")
            return
        
        name = simpledialog.askstring("Guardar Señal de Salida", "Nombre para la señal de salida:", parent=self.root)
        if name:
            if name in self.saved_signals:
                if not messagebox.askyesno("Confirmar", f"Ya existe una señal llamada '{name}'. ¿Sobrescribir?"):
                    return
            self.saved_signals[name] = {'data': self.output_signal.copy(), 'fs': self.fs}
            messagebox.showinfo("Éxito", f"Señal de salida '{name}' guardada en memoria.")
            return True
        return False

    def start_recording(self):
        """Inicia la grabación"""
        if hasattr(self, 'is_recording') and self.is_recording:
            return
        
        self.recorded_frames, self.input_signal = [], None
        self.is_recording = True
        self.is_paused_recording = False
        
        try:
            def _rec_callback(indata, frames, time, status):
                if self.is_recording and not self.is_paused_recording:
                    self.recorded_frames.append(indata.copy())
            
            self.recording_stream = sd.InputStream(samplerate=self.fs, channels=1, callback=_rec_callback)
            self.recording_stream.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar la grabación:\n{e}")
            print(f"Error en grabación: {traceback.format_exc()}")
            self.is_recording = False

    def pause_recording(self):
        """Pausa o reanuda la grabación"""
        if not hasattr(self, 'is_recording') or not self.is_recording:
            return
        
        self.is_paused_recording = not self.is_paused_recording

    def stop_recording(self):
        """Detiene la grabación de manera segura"""
        if not hasattr(self, 'is_recording') or not self.is_recording:
            return
        
        print("Deteniendo grabación...")
        self.is_recording = False
        
        # Cerrar el stream de grabación de manera segura
        if hasattr(self, 'recording_stream') and self.recording_stream:
            try:
                # Intentar detener de manera ordenada primero
                try:
                    self.recording_stream.stop()
                except Exception as e:
                    print(f"Error al detener stream: {e}")
                
                # Cerrar el stream
                try:
                    self.recording_stream.close()
                except Exception as e:
                    print(f"Error al cerrar stream: {e}")
                    
            except Exception as e:
                print(f"Error crítico al cerrar stream de grabación: {e}")
            finally:
                self.recording_stream = None
        
        # Procesar los frames grabados si hay datos
        if hasattr(self, 'recorded_frames') and self.recorded_frames:
            try:
                self.input_signal = np.concatenate(self.recorded_frames, axis=0).flatten()
                duration = len(self.input_signal) / self.fs
                print(f"Grabación completada: {duration:.2f}s, {len(self.input_signal)} muestras")
                
                # Actualizar la interfaz de usuario
                if hasattr(self, 'status_label'):
                    self.status_label.config(text=f"Grabación finalizada ({duration:.2f}s)", foreground="blue")
                    
            except Exception as e:
                error_msg = f"No se pudo procesar la grabación: {str(e)}"
                print(error_msg)
                if hasattr(self, 'status_label'):
                    self.status_label.config(text="Error al procesar grabación", foreground="red")
                messagebox.showerror("Error", error_msg)
        
        # Restaurar estado de botones de manera segura
        try:
            if hasattr(self, 'record_btn') and self.record_btn.winfo_exists():
                self.record_btn.config(state=tk.NORMAL)
            if hasattr(self, 'pause_btn') and self.pause_btn.winfo_exists():
                self.pause_btn.config(state=tk.DISABLED, text="|| Pausa")
            if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                self.stop_btn.config(state=tk.DISABLED)
            if hasattr(self, 'play_btn') and self.play_btn.winfo_exists():
                self.play_btn.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Error al actualizar botones: {e}")
            
    def _save_signal(self, signal_data, default_name):
        """Guarda una señal en un archivo WAV"""
        if signal_data is None:
            messagebox.showwarning("Sin señal", "No hay señal para guardar.")
            return
        
        filename = filedialog.asksaveasfilename(
            initialdir=self.signals_dir,
            title="Guardar señal",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=f"{default_name}.wav"
        )
        
        if filename:
            try:
                # Normalizar la señal para evitar clipping
                signal_normalized = signal_data / np.max(np.abs(signal_data)) * 0.8
                sf.write(filename, signal_normalized, self.fs)
                messagebox.showinfo("Éxito", f"Señal guardada en {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la señal: {str(e)}")
                print(f"Error al guardar señal: {traceback.format_exc()}")

    def cleanup_audio_resources(self):
        """Limpia todos los recursos de audio de manera segura"""
        print("Limpiando recursos de audio...")
        
        # Detener y cerrar todos los streams
        streams_to_close = [
            'realtime_stream', 'playback_stream', 'recording_stream'
        ]
        
        for stream_name in streams_to_close:
            if hasattr(self, stream_name) and getattr(self, stream_name) is not None:
                try:
                    stream = getattr(self, stream_name)
                    if hasattr(stream, 'active') and stream.active:
                        try:
                            stream.stop()
                        except Exception as e:
                            print(f"Error al detener {stream_name}: {e}")
                    
                    time.sleep(0.05)  # Pequeña pausa
                    
                    try:
                        stream.close()
                    except Exception as e:
                        print(f"Error al cerrar {stream_name}: {e}")
                        
                except Exception as e:
                    print(f"Error crítico con {stream_name}: {e}")
                finally:
                    setattr(self, stream_name, None)
        
        # Limpiar buffers
        if hasattr(self, 'recorded_frames'):
            self.recorded_frames = []
        
        # Resetear estados
        self.is_recording = False
        self.is_paused_recording = False
        self.is_playing_recording = False
        self.is_paused_playback = False
        self.playback_position = 0
        
        print("Recursos de audio limpiados")
    
    ## --------------------------------------------------------------------------
    ## 5. MÉTODOS DE DISEÑO DE FILTROS (CON MODO BÁSICO/AVANZADO)
    ## --------------------------------------------------------------------------
    
    def _design_filter(self, is_advanced, filter_frame):
        """Diseña el filtro según el modo seleccionado (básico o avanzado)"""
        if is_advanced:
            self._run_in_thread(lambda: self._design_advanced_filter(filter_frame), 
                               lambda result: self._on_design_complete(result))
        else:
            self._run_in_thread(lambda: self._design_basic_filter(filter_frame), 
                               lambda result: self._on_design_complete(result))
    
    def _design_basic_filter(self, filter_frame):
            """
            Diseña filtro con estabilidad numérica mejorada
            """
            try:
                clase = filter_frame.bf_class.get()
                efecto = filter_frame.bf_effect.get()
                orden = int(filter_frame.bf_order.get())
                fc_low = float(filter_frame.bf_fc.get())
                
                #   CALCULO CORRECTO de Wn
                nyquist = self.fs / 2.0
                
                # Validar frecuencia de corte
                if fc_low <= 0 or fc_low >= nyquist:
                    raise ValueError(f"Frecuencia de corte debe estar entre 0 y {nyquist} Hz")
                
                if efecto in ["Pasa-Banda", "Rechaza-Banda"]:
                    fc_high = float(filter_frame.bf_bw.get())
                    # VERIFICAR que fc_low < fc_high
                    if fc_low >= fc_high:
                        raise ValueError(f"Frecuencia inferior ({fc_low} Hz) debe ser MENOR que superior ({fc_high} Hz)")
                    if fc_high >= nyquist:
                        raise ValueError(f"Frecuencia superior ({fc_high} Hz) debe ser MENOR que Nyquist ({nyquist} Hz)")
                    
                    #   NORMALIZACIÓN CORRECTA
                    Wn = [fc_low/nyquist, fc_high/nyquist]
                else:
                    #   NORMALIZACIÓN CORRECTA para pasa-bajas/pasa-altas
                    Wn = fc_low/nyquist
                
                # Diseñar filtro con parámetros más conservadores
                if clase == "IIR":
                    #   PARA IIR: Usar orden más bajo y sosfilt para estabilidad
                    if orden > 8:  # Limitar orden para IIR
                        orden = 8
                        print(f"  WARNING: Orden reducido a {orden} para estabilidad IIR")
                    
                    #   USAR signal.iirfilter en lugar de butter directamente
                    if efecto == "Pasa-Bajas":
                        b, a = signal.butter(orden, Wn, btype='lowpass')
                    elif efecto == "Pasa-Altas":
                        b, a = signal.butter(orden, Wn, btype='highpass') 
                    elif efecto == "Pasa-Banda":
                        b, a = signal.butter(orden, Wn, btype='bandpass')
                    elif efecto == "Rechaza-Banda":
                        b, a = signal.butter(orden, Wn, btype='bandstop')
                        
                    #   VERIFICAR ESTABILIDAD
                    if not self._es_filtro_estable(b, a):
                        print("  WARNING: Filtro IIR inestable, reduciendo orden...")
                        # Reducir orden hasta que sea estable
                        for orden_reducido in range(orden-1, 1, -1):
                            if efecto == "Pasa-Bajas":
                                b, a = signal.butter(orden_reducido, Wn, btype='lowpass')
                            elif efecto == "Pasa-Altas":
                                b, a = signal.butter(orden_reducido, Wn, btype='highpass')
                            elif efecto == "Pasa-Banda":
                                b, a = signal.butter(orden_reducido, Wn, btype='bandpass')
                            elif efecto == "Rechaza-Banda":
                                b, a = signal.butter(orden_reducido, Wn, btype='bandstop')
                            
                            if self._es_filtro_estable(b, a):
                                print(f"  Filtro estable con orden {orden_reducido}")
                                break
                        else:
                            raise ValueError("No se pudo diseñar un filtro IIR estable")
                            
                else:  # FIR
                    #   CONFIGURACIÓN CORRECTA de pass_zero para FIR
                    pass_zero_config = {
                        "Pasa-Bajas": True,
                        "Pasa-Altas": False, 
                        "Pasa-Banda": False,
                        "Rechaza-Banda": True
                    }
                    
                    # Para FIR, usar orden más alto para mejor respuesta
                    if orden < 50:
                        orden = 50
                        print(f"  WARNING: Orden aumentado a {orden} para mejor respuesta FIR")
                    
                    b = signal.firwin(orden+1, Wn, pass_zero=pass_zero_config[efecto], fs=self.fs)
                    a = np.array([1.0])
                
                print(f"  Filtro {efecto} {clase} diseñado: orden={len(b)-1 if clase=='FIR' else orden}")
                return {'b': b, 'a': a, 'type': clase, 'effect': efecto}
                
            except Exception as e:
                print(f"   Error en diseño de filtro: {e}")
                return None

    def _design_advanced_filter(self, filter_frame):
        """Diseña un filtro en modo avanzado"""
        try:
            # Obtener parámetros desde el frame
            clase = filter_frame.af_class.get()
            config = filter_frame.af_config.get()
            efecto = filter_frame.af_effect.get()
            orden = int(filter_frame.af_order.get())
            fc_low = float(filter_frame.af_fc_low.get())
            
            # Validar parámetros básicos
            if clase == "FIR" and (orden < 20 or orden > 10000):
                raise ValueError("El orden para filtros FIR debe estar entre 100 and 10000")
            if clase == "IIR" and (orden < 1 or orden > 15):
                raise ValueError("El orden para filtros IIR debe estar entre 1 and 15")
            if fc_low <= 0 or fc_low >= self.fs/2:
                raise ValueError(f"La frecuencia de corte debe estar entre 0 y {self.fs/2} Hz")
            
            # Configurar parámetros según tipo de filtro
            nyq = 0.5 * self.fs
            
            # Mapear tipo de filtro a parámetros de scipy
            btype_map = {
                "Pasa-Bajas": 'lowpass',
                "Pasa-Altas": 'highpass', 
                "Pasa-Banda": 'bandpass',
                "Rechaza-Banda": 'bandstop'
            }
            
            btype = btype_map[efecto]
            
            # Configurar frecuencias según tipo de filtro
            if efecto in ["Pasa-Banda", "Rechaza-Banda"]:
                fc_high = float(filter_frame.af_fc_high.get())
                if fc_high <= fc_low or fc_high >= self.fs/2:
                    raise ValueError(f"La frecuencia superior debe ser mayor que {fc_low} y menor que {self.fs/2} Hz")
                Wn = [fc_low/nyq, fc_high/nyq]
            else:
                Wn = fc_low/nyq
            
            # Diseñar filtro según tipo y configuración
            if clase == "IIR":
                if config == "Butterworth":
                    b, a = signal.butter(orden, Wn, btype=btype)
                elif config == "Chebyshev I":
                    rp = float(filter_frame.af_ripple.get())
                    if rp <= 0:
                        raise ValueError("El ripple en banda de paso debe ser mayor que 0")
                    b, a = signal.cheby1(orden, rp, Wn, btype=btype)
                elif config == "Chebyshev II":
                    rs = float(filter_frame.af_stopband.get())
                    if rs <= 0:
                        raise ValueError("El ripple en banda de rechazo debe ser mayor que 0")
                    b, a = signal.cheby2(orden, rs, Wn, btype=btype)
                elif config == "Elíptica":
                    rp = float(filter_frame.af_ripple.get())
                    rs = float(filter_frame.af_stopband.get())
                    if rp <= 0 or rs <= 0:
                        raise ValueError("Los ripples deben ser mayores que 0")
                    b, a = signal.ellip(orden, rp, rs, Wn, btype=btype)
            else:  # FIR - diferentes tipos de ventana
                # Determinar la ventana a usar
                if config == "Ventana Kaiser":
                    # Kaiser con beta óptimo
                    beta = 8  # Valor típico para Kaiser
                    window = ('kaiser', beta)
                elif config == "Ventana Hamming":
                    window = 'hamming'
                elif config == "Ventana Blackman":
                    window = 'blackman'
                elif config == "Ventana Rectangular":
                    window = 'boxcar'
                else:
                    window = 'hamming'  # Por defecto
                
                # Diseñar filtro FIR
                if efecto in ["Pasa-Banda", "Rechaza-Banda"]:
                    b = signal.firwin(orden+1, Wn, window=window, pass_zero=(btype == 'bandstop'))
                else:
                    b = signal.firwin(orden+1, Wn, window=window, pass_zero=(btype == 'lowpass'))
                a = np.array([1.0])
            
            return {'b': b, 'a': a, 'type': clase, 'config': config, 'effect': efecto, 'advanced': True}
            
        except ValueError as e:
            messagebox.showerror("Error de parámetros", str(e))
            return None
        except Exception as e:
            messagebox.showerror("Error de diseño", f"No se pudo diseñar el filtro: {str(e)}")
            print(f"Error en diseño avanzado: {traceback.format_exc()}")
            return None

    def _on_design_complete(self, result):
        """Maneja la finalización del diseño del filtro"""
        if result is not None:
            self.current_filter = result
            messagebox.showinfo("Éxito", "Filtro diseñado correctamente.")
    
    def save_filter(self):
        """Guarda el filtro actual en un archivo JSON"""
        if self.current_filter is None:
            messagebox.showwarning("Sin filtro", "No hay filtro para guardar.")
            return
        
        filename = filedialog.asksaveasfilename(
            initialdir=self.filters_dir,
            title="Guardar filtro",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                # Crear copia del filtro para guardar (sin posibles objetos no serializables)
                filter_data = {
                    'b': self.current_filter['b'].tolist(),
                    'a': self.current_filter['a'].tolist(),
                    'type': self.current_filter.get('type', 'IIR'),
                    'effect': self.current_filter.get('effect', 'Pasa-Bajas'),
                    'fs': self.fs
                }
                
                with open(filename, 'w') as f:
                    json.dump(filter_data, f, indent=4)
                
                messagebox.showinfo("Éxito", f"Filtro guardado en {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el filtro: {str(e)}")
                print(f"Error al guardar filtro: {traceback.format_exc()}")
    
    def load_filter(self):
        """Carga un filtro desde un archivo JSON"""
        filename = filedialog.askopenfilename(
            initialdir=self.filters_dir,
            title="Cargar filtro",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    filter_data = json.load(f)
                
                # Convertir listas de vuelta a arrays numpy
                self.current_filter = {
                    'b': np.array(filter_data['b']),
                    'a': np.array(filter_data['a']),
                    'type': filter_data.get('type', 'IIR'),
                    'effect': filter_data.get('effect', 'Pasa-Bajas')
                }
                
                # Actualizar frecuencia de muestreo si está guardada en el filtro
                if 'fs' in filter_data:
                    self.fs = filter_data['fs']
                    self._update_fs_display()
                
                messagebox.showinfo("Éxito", f"Filtro cargado desde {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el filtro: {str(e)}")
                print(f"Error al cargar filtro: {traceback.format_exc()}")

    ## --------------------------------------------------------------------------
    ## 6. MÉTODOS DE SALIDA Y APLICACIÓN DE FILTROS (CON DOUBLE CONTROL DE REPRODUCCIÓN)
    ## --------------------------------------------------------------------------
    
    def _apply_filter_thread(self, signal_name, filter_name):
        """Aplica el filtro en un hilo separado"""
        if not signal_name or not filter_name:
            messagebox.showwarning("Selección requerida", "Debes seleccionar una señal y un filtro.")
            return
        
        self._run_in_thread(
            target_func=lambda: self._apply_filter_task(signal_name, filter_name),
            callback_success=self._on_filter_applied
        )

    def _apply_filter_task(self, signal_name, filter_name):
        """Tarea para aplicar el filtro"""
        input_data, fs = self.saved_signals[signal_name]['data'], self.saved_signals[signal_name]['fs']
        with open(os.path.join(self.filters_dir, filter_name), 'r') as f:
            filter_info = json.load(f)
        b, a = np.array(filter_info['b']), np.array(filter_info['a'])
        output_data = signal.lfilter(b, a, input_data)
        return input_data, output_data, fs

    def _on_filter_applied(self, result):
        """Maneja la finalización de la aplicación del filtro"""
        self.input_signal, self.output_signal, self.fs = result
        messagebox.showinfo("Éxito", "Filtro aplicado correctamente.")

    # --------------------------------------------------------------------------
    # 7. MÉTODOS DE PROCESAMIENTO EN TIEMPO REAL
    # --------------------------------------------------------------------------
    
    def start_realtime_processing(self):
        """Inicia el procesamiento en tiempo real optimizado para RPi"""
        if self.current_filter is None:
            messagebox.showwarning("Sin Filtro", "Primero debes diseñar o cargar un filtro en la Etapa 2.")
            return
        if self.realtime_stream and self.realtime_stream.active:
            return

        try:
            # Prepara el estado inicial del filtro
            b = self.current_filter['b']
            a = self.current_filter['a']
            
            # Precomputar para mejor rendimiento
            if self.current_filter['type'] == 'IIR':
                self.realtime_filter_zi = signal.lfilter_zi(b, a)
            else:
                self.realtime_filter_zi = None

            # Buffer para reducir la frecuencia de actualización de gráficos
            self.plot_buffer_size = 512  # Reducido para RPi
            self.in_buffer = deque(maxlen=self.plot_buffer_size)
            self.out_buffer = deque(maxlen=self.plot_buffer_size)
            
            def callback(indata, outdata, frames, time, status):
                if status:
                    print(f"Stream status: {status}")
                    
                # Filtra el audio entrante
                if self.current_filter['type'] == 'IIR':
                    filtered, self.realtime_filter_zi = signal.lfilter(
                        b, a, indata[:, 0], zi=self.realtime_filter_zi
                    )
                else:
                    filtered = signal.lfilter(b, a, indata[:, 0])
                
                # Asegurar que la salida tenga la forma correcta (frames, 1)
                outdata[:] = filtered.reshape(-1, 1)
                
                # Reducir la frecuencia de muestreo para plotting
                if len(self.in_buffer) < self.plot_buffer_size:
                    self.in_buffer.extend(indata.flatten())
                    self.out_buffer.extend(filtered)
                
                # Enviar datos a la cola solo cuando el buffer esté lleno
                if len(self.in_buffer) == self.plot_buffer_size:
                    try:
                        self.realtime_plot_queue.put_nowait((
                            np.array(self.in_buffer), 
                            np.array(self.out_buffer)
                        ))
                        self.in_buffer.clear()
                        self.out_buffer.clear()
                    except queue.Full:
                        pass  # Ignorar si la cola está llena
                        
            # Configurar stream con buffer más grande para mejor rendimiento
            self.realtime_stream = sd.Stream(
                samplerate=self.fs, 
                channels=1, 
                callback=callback,
                blocksize=1024,  # Tamaño de bloque aumentado
                latency='high'  # Mayor latencia para mejor estabilidad
            )
            self.realtime_stream.start()
            
        except Exception as e:
            messagebox.showerror("Error de Stream", f"No se pudo iniciar el stream de audio:\n{e}")
            print(f"Error en tiempo real: {traceback.format_exc()}")

    def stop_realtime_processing(self):
        """Detiene el procesamiento en tiempo real"""
        if self.realtime_stream:
            try:
                self.realtime_stream.stop()
                self.realtime_stream.close()
            except:
                pass
            self.realtime_stream = None

    # --------------------------------------------------------------------------
    # 8. REPRODUCCIÓN DE AUDIO
    # --------------------------------------------------------------------------
    
    def _create_playback_controls(self, parent_frame, key):
        frame = ttk.Frame(parent_frame)
        
        play_btn = ttk.Button(frame, text="▶", width=3, command=lambda: self.play_pause_audio(key))
        play_btn.pack(side=tk.LEFT, padx=2)
        
        pause_btn = ttk.Button(frame, text="||", width=3, command=self.pause_audio)
        pause_btn.pack(side=tk.LEFT, padx=2)
        
        stop_btn = ttk.Button(frame, text="■", width=3, command=self.stop_audio)
        stop_btn.pack(side=tk.LEFT, padx=2)
        
        if not hasattr(self, 'playback_buttons'): self.playback_buttons = {}
        self.playback_buttons[key] = {'play': play_btn, 'pause': pause_btn, 'stop': stop_btn}
        return frame

    def _update_playback_buttons_state(self, key, enabled):
        """Actualiza el estado de los botones de reproducción con manejo de errores"""
        try:
            state = tk.NORMAL if enabled else tk.DISABLED
            if hasattr(self, 'playback_buttons') and key in self.playback_buttons:
                for btn in self.playback_buttons[key].values():
                    # Verificar si el widget todavía existe antes de intentar configurarlo
                    if btn.winfo_exists():
                        btn.config(state=state)
        except Exception as e:
            # Ignorar errores si los widgets han sido destruidos
            print(f"Error al actualizar botones de reproducción: {e}")

    def play_pause_audio(self, key):
        # Determina qué señal reproducir basado en la clave
        if key == "input" or key == "output_in":
            signal_to_play = self.input_signal
        elif key == "output_out":
            signal_to_play = self.output_signal
        else: return
        
        if signal_to_play is None: return

        # Si ya se está reproduciendo otra cosa, párala
        if self.playback_stream and self.current_playback_signal is not signal_to_play:
            self.stop_audio()
        
        self.current_playback_signal = signal_to_play

        # Reanudar si está en pausa
        if self.is_paused:
            self.is_paused = False
            return
            
        # Iniciar desde el principio si no hay stream
        if not self.playback_stream or not self.playback_stream.active:
            self.playback_position = 0
            self.play_audio(self.current_playback_signal, self.fs)

    def play_audio(self, signal_data, fs):
        try:
            # Normalizar la señal para evitar clipping
            max_val = np.max(np.abs(signal_data))
            if max_val > 0:
                signal_data = signal_data / max_val * 0.8
                
            def callback(outdata, frames, time, status):
                if status:
                    print(f"Playback status: {status}")
                    
                if self.is_paused: 
                    outdata.fill(0)
                    return
                    
                start_pos = self.playback_position
                end_pos = start_pos + frames
                
                if start_pos >= len(signal_data):
                    raise sd.CallbackStop
                    
                chunk = signal_data[start_pos:end_pos]
                
                if len(chunk) < frames:
                    outdata[:len(chunk)] = chunk.reshape(-1, 1)
                    outdata[len(chunk):].fill(0)
                    raise sd.CallbackStop
                else:
                    outdata[:] = chunk.reshape(-1, 1)
                
                self.playback_position = end_pos
                
            # Configurar stream con parámetros optimizados
            self.playback_stream = sd.OutputStream(
                samplerate=int(fs), 
                channels=1, 
                callback=callback, 
                finished_callback=self.stop_audio,
                blocksize=1024,
                latency='high'
            )
            self.playback_stream.start()
            
        except Exception as e: 
            messagebox.showerror("Error", f"No se pudo reproducir: {e}")
            print(f"Error en reproducción: {traceback.format_exc()}")
            self.stop_audio()

    def pause_audio(self):
        self.is_paused = not self.is_paused

    def stop_audio(self):
        """Detiene la reproducción de audio de manera segura"""
        try:
            if self.playback_stream:
                # Verificar si el stream está activo antes de intentar detenerlo
                if hasattr(self.playback_stream, 'active') and self.playback_stream.active:
                    self.playback_stream.stop()
                if hasattr(self.playback_stream, 'close'):
                    self.playback_stream.close()
        except Exception as e:
            # Capturar cualquier excepción para evitar que el programa se cierre
            print(f"Error al detener reproducción: {e}")
        finally:
            # Resetear estados independientemente de si hubo error o no
            self.playback_stream = None
            self.is_paused = False
            self.playback_position = 0
            self.current_playback_signal = None
            
            # Actualizar botones de reproducción solo si existen
            try:
                if hasattr(self, 'playback_buttons'):
                    for key in self.playback_buttons:
                        # Verificar si los botones todavía existen antes de intentar actualizarlos
                        if all(btn.winfo_exists() for btn in self.playback_buttons[key].values()):
                            self._update_playback_buttons_state(key, self.input_signal is not None or 
                                                               (hasattr(self, 'output_signal') and self.output_signal is not None))
            except Exception as e:
                print(f"Error al actualizar botones después de detener: {e}")
    
    # --------------------------------------------------------------------------
    # 9. PLOTTING (GRAFICACIÓN) REFACTORIZADO CON CONTROLES SUPERIORES
    # --------------------------------------------------------------------------

    def show_signal_plot_window(self, signal_data, fs, title, parent_frame=None):
        """Muestra ventana de gráficas con controles de ejes en la parte superior - PARA TODAS LAS GRÁFICAS"""
        if signal_data is None or not self.signal_loaded: 
            messagebox.showwarning("Sin Datos", "No hay señal para graficar.")
            return
        
        # Crear ventana nueva para gráficas
        def create_plot_window(window):
            self._create_signal_plots_in_frame(signal_data, fs, title, window)
        
        self._show_window(f"Gráfica: {title}", create_plot_window)

    def _create_signal_plots_in_frame(self, signal_data, fs, title, parent_frame):
        """Crea gráficas con controles de ejes en la parte superior del frame proporcionado"""
        # Frame principal
        main_frame = ttk.Frame(parent_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ========== CONTROLES SUPERIORES ==========
        controls_frame = ttk.LabelFrame(main_frame, text="Configuración de Ejes", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Variables para unidades (usar las variables globales de la app)
        time_unit_var = tk.StringVar(value=self.time_units.get())
        freq_unit_var = tk.StringVar(value=self.freq_units.get())
        mag_unit_var = tk.StringVar(value=self.mag_units.get())
        
        # Fila 1: Unidades de ejes
        units_frame = ttk.Frame(controls_frame)
        units_frame.pack(fill=tk.X, pady=5)
        
        # Controles de tiempo
        ttk.Label(units_frame, text="Eje Tiempo:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        time_units_cb = ttk.Combobox(units_frame, textvariable=time_unit_var, 
                                    values=["segundos", "muestras"], width=12, state="readonly")
        time_units_cb.pack(side=tk.LEFT, padx=5)
        
        # Controles de frecuencia
        ttk.Label(units_frame, text="Eje Frecuencia:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=15)
        freq_units_cb = ttk.Combobox(units_frame, textvariable=freq_unit_var, 
                                    values=["Hz", "rad/muestra"], width=12, state="readonly")
        freq_units_cb.pack(side=tk.LEFT, padx=5)
        
        # Controles de magnitud
        ttk.Label(units_frame, text="Eje Magnitud (Y):", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=15)
        mag_units_cb = ttk.Combobox(units_frame, textvariable=mag_unit_var, 
                                values=["Adimensional", "dB"], width=15, state="readonly")
        mag_units_cb.pack(side=tk.LEFT, padx=5)
        
        # ========== GRÁFICAS ==========
        # Crear figura con más espacio entre subplots
        fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.4, top=0.92, bottom=0.08, left =12, right=95)  # Ajuste de espacio en las gráficas
        
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barra de herramientas
        toolbar = NavigationToolbar2Tk(canvas, main_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        def update_plot(*args):
            """Actualiza la gráfica con las unidades seleccionadas"""
            ax_time.clear()
            ax_freq.clear()
            
            # Graficar con las unidades seleccionadas
            self._plot_signal_with_units(ax_time, ax_freq, signal_data, fs, title,
                                    time_unit_var.get(), freq_unit_var.get(), mag_unit_var.get())
            
            # Asegurar que los títulos no se traslapen
            ax_time.set_title(f"{title}: Dominio de Tiempo", fontweight='bold', fontsize=10, pad=15)
            ax_freq.set_title(f"{title}: Espectro de Frecuencia", fontweight='bold', fontsize=10, pad=15)
            
            canvas.draw()
        
        def on_units_change(*args):
            """Callback para cambios en las unidades"""
            update_plot()
        
        # Configurar actualización automática
        for var in [time_unit_var, freq_unit_var, mag_unit_var]:
            var.trace('w', on_units_change)
        
        # Actualizar gráfica inicialmente
        update_plot()

    def _plot_signal_with_units(self, ax_time, ax_freq, signal_data, fs, title, 
                        time_unit, freq_unit, mag_unit):
        """Grafica señal con unidades específicas - SIN LÍMITES"""
        if signal_data is None or len(signal_data) == 0: 
            return
                
        # Limitar longitud pero mantener definición
        signal_to_plot = self.limit_signal_length(signal_data, fs, max_duration=30)
        
        # 1. DOMINIO DE TIEMPO - SIN LÍMITES
        if time_unit == "segundos":
            time_vec = np.arange(len(signal_to_plot)) / fs
            xlabel_time = "Tiempo (s)"
        else:
            time_vec = np.arange(len(signal_to_plot))
            xlabel_time = "Muestras"
        
        ax_time.plot(time_vec, signal_to_plot, 'b-', linewidth=1.5)
        ax_time.set_title(f"{title}: Dominio de Tiempo", fontweight='bold', fontsize=10, pad=15)  
        ax_time.set_xlabel(xlabel_time, fontsize=8, labelpad=10)
        ax_time.set_ylabel("Amplitud", fontsize=8, labelpad=10)
        ax_time.grid(True, alpha=0.4, linestyle='--')
        ax_time.autoscale(enable=True, axis='both')
        
        # 2. DOMINIO DE FRECUENCIA 
        freq_pos, magnitude = self.compute_magnitude_spectrum(
            signal_data, fs, db_scale=(mag_unit == "dB")
        )

        if freq_pos is not None and magnitude is not None:
            # Verificar que las frecuencias estén en el rango 
            print(f" Gráfica frecuencia: {len(freq_pos)} puntos, {freq_pos[0]:.1f} a {freq_pos[-1]:.1f} Hz")
            
            # Manejo de unidades de frecuencia
            if freq_unit == "rad/muestra":
                # Convertir Hz a rad/muestra
                freq_pos = 2 * np.pi * freq_pos / fs
                xlabel_freq = "Frecuencia (rad/muestra)"
                xlim_freq = (0, np.pi)
            elif freq_unit == "rad/s":
                # Convertir Hz a rad/s
                freq_pos = 2 * np.pi * freq_pos
                xlabel_freq = "Frecuencia (rad/s)"
                xlim_freq = (0, np.pi * fs)
            else:  # Hz - ESTA ES LA OPCIÓN POR DEFECTO
                xlabel_freq = "Frecuencia (Hz)"
                # CORRECCIÓN: Límites de 0 a Fs/2
                xlim_freq = (0, fs/2)
            
            ax_freq.plot(freq_pos, magnitude, 'r-', linewidth=1.5)
            ax_freq.set_title(f"{title}: Espectro de Frecuencia", fontweight='bold', fontsize=10, pad=15)
            ax_freq.set_xlabel(xlabel_freq, fontsize=8)
            ax_freq.set_xlim(xlim_freq)  # ESTABLECER LÍMITES CORRECTOS
            
            # CORRECCIÓN: Etiquetas de magnitud
            if mag_unit == "dB":
                ax_freq.set_ylabel("Magnitud (dB)", fontsize=8, labelpad=10)
            else:
                ax_freq.set_ylabel("Magnitud", fontsize=8, labelpad=10)
            
            ax_freq.grid(True, alpha=0.3, linestyle='--')
            
            # SIN LÍMITES - escala automática
            ax_freq.autoscale(enable=True, axis='both')
        
        # Mejorar aspecto con más espacio
        for ax in [ax_time, ax_freq]:
            ax.tick_params(labelsize=8, pad=8)  # Añadir padding a los ticks
            ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))  
            
        # Ajuste automático del layout
        plt.tight_layout()      
    
    def _reset_plot_view(self, ax_time, ax_freq):
        """Resetea la vista de las gráficas a automático"""
        ax_time.relim()
        ax_time.autoscale_view()
        ax_freq.relim()
        ax_freq.autoscale_view()
        self.time_scale_var.set("auto")
        self.freq_scale_var.set("auto")

    def show_frequency_response_window(self):
        """Muestra la ventana de respuesta en frecuencia CON CONTROLES SUPERIORES"""
        
        if self.current_filter is None: 
            messagebox.showwarning("Sin Filtro", "Primero debes diseñar o cargar un filtro.")
            return
        
        def create_response_window(window):
            # Frame principal
            main_frame = ttk.Frame(window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # ========== CONTROLES SUPERIORES ==========
            controls_frame = ttk.LabelFrame(main_frame, text="Configuración de Ejes - Respuesta en Frecuencia", padding=10)
            controls_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Variables para unidades
            freq_unit_var = tk.StringVar(value="Hz")
            mag_unit_var = tk.StringVar(value="dB")
            phase_unit_var = tk.StringVar(value="grados")
            
            # Fila 1: Unidades de ejes
            units_frame = ttk.Frame(controls_frame)
            units_frame.pack(fill=tk.X, pady=5)
            
            # Controles de frecuencia
            ttk.Label(units_frame, text="Eje Frecuencia (X):", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
            freq_units = ttk.Combobox(units_frame, textvariable=freq_unit_var,
                                    values=["Hz", "rad/muestra", "rad/s"], 
                                    width=12, state="readonly")
            freq_units.pack(side=tk.LEFT, padx=5)
            
            # Controles de magnitud
            ttk.Label(units_frame, text="Eje Magnitud (Y):", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=15)
            mag_units = ttk.Combobox(units_frame, textvariable=mag_unit_var,
                                    values=["Lineal", "dB"], 
                                    width=10, state="readonly")
            mag_units.pack(side=tk.LEFT, padx=5)
            
            # Controles de fase
            ttk.Label(units_frame, text="Eje Fase (Y):", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=15)
            phase_units = ttk.Combobox(units_frame, textvariable=phase_unit_var,
                                    values=["grados", "radianes"], 
                                    width=10, state="readonly")
            phase_units.pack(side=tk.LEFT, padx=5)
            
            # ========== GRÁFICAS ==========
            # Crear figura con 2 subplots (magnitud y fase) con más espacio
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8))
            fig.subplots_adjust(hspace=0.5, top=0.92, bottom=0.08)  
            
            canvas = FigureCanvasTkAgg(fig, master=main_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Barra de herramientas
            toolbar = NavigationToolbar2Tk(canvas, main_frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            def update_plot(*args):
                """Actualiza la gráfica con las unidades seleccionadas"""
                ax_mag.clear()
                ax_phase.clear()
                
                # Obtener unidades seleccionadas
                freq_units_val = freq_unit_var.get()
                mag_units_val = mag_unit_var.get()
                phase_units_val = phase_unit_var.get()
                
                # Calcular respuesta en frecuencia
                w, h = signal.freqz(self.current_filter['b'], self.current_filter['a'], worN=8000)
                
                # Convertir unidades de frecuencia
                if freq_units_val == "Hz":
                    freq = (w / np.pi) * (self.fs / 2)
                    freq_label = "Frecuencia (Hz)"
                    # Limitar a frecuencia de Nyquist
                    freq = np.clip(freq, 0, self.fs/2)
                elif freq_units_val == "rad/muestra":
                    freq = w
                    freq_label = "Frecuencia (rad/muestra)"
                else:  # rad/s
                    freq = w * self.fs
                    freq_label = "Frecuencia (rad/s)"
                
                # Convertir unidades de magnitud
                magnitude = np.abs(h)
                if mag_units_val == "dB":
                    magnitude = 20 * np.log10(np.maximum(magnitude, 1e-10))
                    mag_label = "Magnitud (dB)"
                else:  # Lineal
                    mag_label = "Magnitud"
                
                # Convertir unidades de fase
                phase = np.unwrap(np.angle(h))
                if phase_units_val == "grados":
                    phase = np.degrees(phase)
                    phase_label = "Fase (grados)"
                else:  # radianes
                    phase_label = "Fase (radianes)"
                
                # Graficar magnitud
                ax_mag.plot(freq, magnitude, 'b-', linewidth=1.5)
                ax_mag.set_title("Respuesta de Magnitud del Filtro", fontweight='bold', fontsize=10, pad=15)
                ax_mag.set_xlabel(freq_label, fontsize=8, labelpad=10)
                ax_mag.set_ylabel(mag_label, fontsize=8, labelpad=10)
                ax_mag.grid(True, alpha=0.3, linestyle='--')
                # SIN LÍMITES - escala automática
                ax_mag.autoscale(enable=True, axis='both')
                
                # Graficar fase
                ax_phase.plot(freq, phase, 'r-', linewidth=1.5)
                ax_phase.set_title("Respuesta de Fase del Filtro", fontweight='bold', fontsize=10, pad=15)
                ax_phase.set_xlabel(freq_label, fontsize=8, labelpad=10)
                ax_phase.set_ylabel(phase_label, fontsize=8, labelpad=10)
                ax_phase.grid(True, alpha=0.3, linestyle='--')
                ax_phase.autoscale(enable=True, axis='both')
                
                # Mejorar aspecto
                for ax in [ax_mag, ax_phase]:
                    ax.tick_params(labelsize=8, pad=8)
                    ax.grid(True, which='minor', alpha=0.2, linestyle=':')
                
                plt.tight_layout()
                canvas.draw()
            
            def on_units_change(*args):
                """Callback para cambios en las unidades"""
                update_plot()
            
            # Configurar actualización automática
            for var in [freq_unit_var, mag_unit_var, phase_unit_var]:
                var.trace('w', on_units_change)
            
            # Actualizar gráfica inicialmente
            update_plot()
        
        self._show_window("Respuesta en Frecuencia del Filtro", create_response_window)

    def _reset_freq_response_view(self, ax_mag, ax_phase):
        """Resetea la vista de las gráficas de respuesta en frecuencia"""
        ax_mag.relim()
        ax_mag.autoscale_view()
        ax_phase.relim()
        ax_phase.autoscale_view()
        self.mag_scale_var.set("auto")
        self.phase_scale_var.set("auto")

    def show_comparison_window(self):
        """Muestra ventana de comparación de señales con límites específicos"""
        if self.input_signal is None or self.output_signal is None:
            messagebox.showwarning("Datos incompletos", "Se necesitan ambas señales (entrada y salida) para comparar.")
            return
        
        def create_comparison_window(window):
            # Frame principal
            main_frame = ttk.Frame(window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # ========== CONTROLES SUPERIORES ==========
            controls_frame = ttk.LabelFrame(main_frame, text="Configuración de Ejes - Comparación", padding=10)
            controls_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Variables para unidades
            time_unit_var = tk.StringVar(value="segundos")
            freq_unit_var = tk.StringVar(value="Hz")
            mag_unit_var = tk.StringVar(value="Adimensional")
            
            # Fila 1: Unidades
            units_frame = ttk.Frame(controls_frame)
            units_frame.pack(fill=tk.X, pady=5)
            
            # Controles de tiempo
            ttk.Label(units_frame, text="Eje Tiempo:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
            time_units = ttk.Combobox(units_frame, textvariable=time_unit_var,
                                    values=["segundos", "muestras"], width=12, state="readonly")
            time_units.pack(side=tk.LEFT, padx=5)
            
            # Controles de frecuencia
            ttk.Label(units_frame, text="Eje Frecuencia:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=15)
            freq_units = ttk.Combobox(units_frame, textvariable=freq_unit_var,
                                    values=["Hz", "rad/muestra"], width=12, state="readonly")
            freq_units.pack(side=tk.LEFT, padx=5)
            
            # Controles de magnitud
            ttk.Label(units_frame, text="Eje Magnitud:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=15)
            mag_units = ttk.Combobox(units_frame, textvariable=mag_unit_var,
                                    values=["Adimensional", "dB"], width=10, state="readonly")
            mag_units.pack(side=tk.LEFT, padx=5)
            
            # ========== GRÁFICAS ==========
            # Crear figura con 2x2 subplots
            fig, ((ax_in_time, ax_out_time), (ax_in_freq, ax_out_freq)) = plt.subplots(2, 2, figsize=(12, 9))
            fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.08)
            fig.suptitle("Comparación de Señales - Entrada vs Salida", fontsize=11, fontweight='bold')
            
            canvas = FigureCanvasTkAgg(fig, master=main_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Barra de herramientas
            toolbar = NavigationToolbar2Tk(canvas, main_frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            def update_plots(*args):
                """Actualiza todas las gráficas con límites específicos para comparación"""
                for ax in [ax_in_time, ax_out_time, ax_in_freq, ax_out_freq]:
                    ax.clear()
                
                # Obtener unidades seleccionadas
                t_unit = time_unit_var.get()
                f_unit = freq_unit_var.get()
                m_unit = mag_unit_var.get()
                
                # CALCULAR LÍMITES PARA COMPARACIÓN
                # Dominio del tiempo: 1.1 veces el valor máximo/mínimo
                if self.input_signal is not None and self.output_signal is not None:
                    max_time = max(np.max(np.abs(self.input_signal)), np.max(np.abs(self.output_signal)))
                    min_time = min(np.min(self.input_signal), np.min(self.output_signal))
                    time_ylim = (min_time * 1.1, max_time * 1.1)
                else:
                    time_ylim = None
                
                # GRÁFICAS DE ENTRADA
                # Dominio de tiempo - entrada
                if t_unit == "segundos":
                    time_vec = np.arange(len(self.input_signal)) / self.fs
                    xlabel_time = "Tiempo (s)"
                else:
                    time_vec = np.arange(len(self.input_signal))
                    xlabel_time = "Muestras"
                
                ax_in_time.plot(time_vec, self.input_signal, 'b-', linewidth=1.2, alpha=0.8)
                ax_in_time.set_title("Entrada: Dominio de Tiempo", fontweight='bold', fontsize=11)
                ax_in_time.set_xlabel(xlabel_time, fontsize=8,labelpad=8)
                ax_in_time.set_ylabel("Amplitud", fontsize=8,labelpad=8)
                ax_in_time.grid(True, alpha=0.3, linestyle='--')
                if time_ylim:
                    ax_in_time.set_ylim(time_ylim)
                
                # Dominio de frecuencia - entrada
                freq_in, mag_in = self.compute_magnitude_spectrum(
                    self.input_signal, self.fs, db_scale=(m_unit == "dB")
                )

                if freq_in is not None and mag_in is not None:
                    if f_unit == "rad/muestra":
                        freq_in = 2 * np.pi * freq_in / self.fs
                        xlabel_freq = "Frecuencia (rad/muestra)"
                        xlim_freq = (0, np.pi)
                    else:
                        xlabel_freq = "Frecuencia (Hz)"
                        xlim_freq = (0, self.fs/2)  # LÍMITE CORRECTO
                    
                    ax_in_freq.plot(freq_in, mag_in, 'b-', linewidth=1.2, alpha=0.8)
                    ax_in_freq.set_title("Entrada: Espectro de Frecuencia", fontweight='bold', fontsize=11)
                    ax_in_freq.set_xlabel(xlabel_freq, fontsize=8)
                    ax_in_freq.set_xlim(xlim_freq)  # ESTABLECER LÍMITE
                    
                    # Para frecuencia: 0 a 1.1 veces el máximo
                    if m_unit == "dB":
                        freq_ylim_db = (np.min(mag_in) * 1.1 if np.min(mag_in) < 0 else -5, 
                                    np.max(mag_in) * 1.1 if np.max(mag_in) > 0 else 5)
                        ax_in_freq.set_ylim(freq_ylim_db)
                    else:
                        freq_ylim_linear = (0, np.max(mag_in) * 1.1)
                        ax_in_freq.set_ylim(freq_ylim_linear)
                
                # GRÁFICAS DE SALIDA
                # Dominio de tiempo - salida
                if t_unit == "segundos":
                    time_vec = np.arange(len(self.output_signal)) / self.fs
                    xlabel_time = "Tiempo (s)"
                else:
                    time_vec = np.arange(len(self.output_signal))
                    xlabel_time = "Muestras"
                
                ax_out_time.plot(time_vec, self.output_signal, 'r-', linewidth=1.2, alpha=0.8)
                ax_out_time.set_title("Salida: Dominio de Tiempo", fontweight='bold', fontsize=11)
                ax_out_time.set_xlabel(xlabel_time, fontsize=8,labelpad=8)
                ax_out_time.set_ylabel("Amplitud", fontsize=8,labelpad=8)
                ax_out_time.grid(True, alpha=0.3, linestyle='--')
                
                if time_ylim:
                    ax_out_time.set_ylim(time_ylim)
                
                # Dominio de frecuencia - salida  
                freq_out, mag_out = self.compute_magnitude_spectrum(
                    self.output_signal, self.fs, db_scale=(m_unit == "dB")
                )
                
                if freq_out is not None and mag_out is not None:
                    if f_unit == "rad/muestra":
                        freq_out = 2 * np.pi * freq_out / self.fs
                        xlabel_freq = "Frecuencia (rad/muestra)"
                    else:
                        xlabel_freq = "Frecuencia (Hz)"
                    
                    ax_out_freq.plot(freq_out, mag_out, 'r-', linewidth=1.2, alpha=0.8)
                    ax_out_freq.set_title("Salida: Espectro de Frecuencia", fontweight='bold', fontsize=11)
                    ax_out_freq.set_xlabel(xlabel_freq, fontsize=8)
                    ax_out_freq.set_ylabel("Magnitud (dB)" if m_unit == "dB" else "Magnitud", fontsize=8)
                    ax_out_freq.grid(True, alpha=0.3, linestyle='--')
                    
                    # Para frecuencia: 0 a 1.1 veces el máximo
                    if m_unit == "dB":
                        freq_ylim_db = (np.min(mag_out) * 1.1 if np.min(mag_out) < 0 else -5, 
                                    np.max(mag_out) * 1.1 if np.max(mag_out) > 0 else 5)
                        ax_out_freq.set_ylim(freq_ylim_db)
                    else:
                        freq_ylim_linear = (0, np.max(mag_out) * 1.1)
                        ax_out_freq.set_ylim(freq_ylim_linear)
                
                    
                # Mejorar aspecto de todas las gráficas
                for ax in [ax_in_time, ax_out_time, ax_in_freq, ax_out_freq]:
                    ax.tick_params(labelsize=8, pad=6)
                    ax.grid(True, which='minor', alpha=0.2, linestyle=':')
                
                # Ajustar layout para evitar traslapes
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                canvas.draw()
            
            def on_units_change(*args):
                """Callback para cambios en las unidades"""
                update_plots()
            
            # Configurar actualización automática
            for var in [time_unit_var, freq_unit_var, mag_unit_var]:
                var.trace('w', on_units_change)
            
            # Actualizar gráficas inicialmente
            update_plots()
        
        self._show_window("Comparación de Señales - Entrada vs Salida", create_comparison_window)

    def _reset_comparison_view(self, ax_in_time, ax_out_time, ax_in_freq, ax_out_freq):
        """Resetea la vista de las gráficas de comparación"""
        for ax in [ax_in_time, ax_out_time, ax_in_freq, ax_out_freq]:
            ax.relim()
            ax.autoscale_view()
        self.comp_time_scale_var.set("auto")
        self.comp_freq_scale_var.set("auto")
        
    def _get_filter_cutoff(self):
        """Intenta obtener la frecuencia de corte del filtro actual"""
        try:
            # Esto es un ejemplo - necesitarías almacenar los parámetros del filtro
            # cuando lo diseñas para poder recuperarlos aquí
            if hasattr(self, 'current_filter_params'):
                return self.current_filter_params.get('fc', '1000')
            return "1000"
        except:
            return "1000"

    def _save_frequency_response_plot(self, fig):
        """Guarda la gráfica de respuesta en frecuencia"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            title="Guardar respuesta en frecuencia"
        )
        
        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Gráfica guardada en {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la gráfica: {str(e)}")

    def _show_filter_info(self):
        """Muestra información detallada del filtro"""
        if self.current_filter is None:
            return
        
        info_text = f"INFORMACIÓN DEL FILTRO\n\n"
        info_text += f"Tipo: {self.current_filter.get('type', 'N/A')}\n"
        info_text += f"Efecto: {self.current_filter.get('effect', 'N/A')}\n"
        
        if 'config' in self.current_filter:
            info_text += f"Configuración: {self.current_filter['config']}\n"
        
        if 'advanced' in self.current_filter:
            info_text += f"Modo: {'Avanzado' if self.current_filter['advanced'] else 'Básico'}\n"
        
        b_coeffs = self.current_filter['b']
        a_coeffs = self.current_filter['a']
        
        info_text += f"\nCOEFICIENTES:\n"
        info_text += f"b (numerador): {len(b_coeffs)} coeficientes\n"
        info_text += f"a (denominador): {len(a_coeffs)} coeficientes\n"
        
        if len(b_coeffs) <= 8:
            info_text += f"b: {['%.6f' % x for x in b_coeffs]}\n"
        if len(a_coeffs) <= 8 and not np.array_equal(a_coeffs, [1.0]):
            info_text += f"a: {['%.6f' % x for x in a_coeffs]}\n"
        
        messagebox.showinfo("Información del Filtro", info_text)
            
# ------------------------------------------------------------------------------
# DEFINICIÓN DE FRAMES PARA CADA PANTALLA
# ------------------------------------------------------------------------------
class WelcomeFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de bienvenida."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Frame para el título a la izquierda
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(title_frame, text="Universidad del Valle de Guatemala", 
                font=("Arial", 12, "bold"), foreground="#003366").pack(anchor="w")
        tk.Label(title_frame, text="Diseño e Innovación en Ingeniería 2", 
                font=("Arial", 10), foreground="#666666").pack(anchor="w")
        
        # Frame para el logo a la derecha
        logo_frame = ttk.Frame(header_frame)
        logo_frame.pack(side=tk.RIGHT, padx=10)
        
        # Mostrar logo con ruta completa
        self.show_logo(logo_frame)
        
        # Centro con título y botón
        center_frame = ttk.Frame(self)
        center_frame.pack(expand=True)

        title_label = tk.Label(center_frame, 
            text="Bienvenidos a la Unidad de Procesamiento Digital de Señales", 
            font=("Arial", 14, "bold"), foreground="#003366", wraplength=700, justify=tk.CENTER)
        title_label.pack(pady=10)
        
        # Botón de información sobre la aplicación
        info_btn = ttk.Button(center_frame, text="ℹ️", style='Info.TButton', width=3)
        info_btn.pack(pady=5)
        ToolTip(info_btn, "Sistema de procesamiento digital de señales \npara análisis y filtrado de audio.\n"
                "Incluye herramientas para entrada de señal, \ndiseño de filtros y análisis espectral.")
        
        start_button = ttk.Button(center_frame, text="COMENZAR", 
                                style='Large.TButton', command=self.app.show_mode_selection)
        start_button.pack(pady=20)

    def show_logo(self, parent_frame):
        """Muestra el logo de la UVG en el frame especificado."""
        try:
            # Usar la ruta completa del archivo
            logo_path = r"/home/pi/Desktop/logo_uvg.png"
            logo_image = tk.PhotoImage(file=logo_path)
            
            # Redimensionar si es necesario (opcional)
            logo_image = logo_image.subsample(2, 2)  # Reducir a la mitad
            
            logo_label = tk.Label(parent_frame, image=logo_image)
            logo_label.image = logo_image  # Mantener referencia
            logo_label.pack(padx=10)
            print("Logo cargado exitosamente")
            
        except Exception as e:
            print(f"No se pudo cargar el logo: {e}")
            # Fallback: crear un logo simple con texto
            logo_label = tk.Label(parent_frame, text="UVG", 
                                 font=("Arial", 12, "bold"), 
                                 foreground="#116600",
                                 background="#F0F0F0",
                                 relief="raised",
                                 padx=10,
                                 pady=5)
            logo_label.pack(padx=10)

class ModeSelectionFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de selección de modo."""
        # Botón de volver
        ttk.Button(self, text="←", command=lambda: self.app._go_back(check_unsaved=False)).place(x=10, y=10)
        
        title_label = tk.Label(self, text="Seleccione el modo de procesamiento", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Frame principal para los botones de imágenes
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=10, expand=True)
        
        # Frame para los botones de imágenes (uno al lado del otro)
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(pady=10)
        
        # Botón Procesamiento por Etapas
        self.create_image_button(
            images_frame, 
            "Procesamiento por Etapas", 
            r"/home/pi/Desktop/etapas_icon.png",
            self.app.show_stages_selection,
            "Procesamiento paso a paso: \nEntrada → Filtro → Salida\n\n"
            "Permite cargar o grabar una señal, \ndiseñar un filtro y aplicar "
            "\nel procesamiento de manera secuencial \npara análisis detallado.",
            side=tk.LEFT,
            padx=40
        )
        
        # Botón Tiempo Real
        self.create_image_button(
            images_frame, 
            "Procesamiento en Tiempo Real", 
            r"/home/pi/Desktop/tiempo_real_icon.jpg",
            self.app.show_realtime_processing,
            "Procesamiento en tiempo \nreal de señal\n\n"
            "Aplica filtros en tiempo \nreal a la señal \nde entrada, mostrando\n "
            "las señales original y\n filtrada simultáneamente.",
            side=tk.RIGHT,
            padx=40
        )

    def create_image_button(self, parent, title, image_path, command, tooltip_text, side=tk.LEFT, padx=20):
        """Crea un botón con imagen y botón de información."""
        container = ttk.Frame(parent)
        container.pack(side=side, padx=padx, pady=10)
        
        # Título arriba del botón de imagen
        title_label = tk.Label(container, text=title, 
                              font=("Arial", 12, "bold"), foreground="#003366")
        title_label.pack(pady=(0, 10))
        
        # Frame para el botón de imagen y el botón de información
        inner_frame = ttk.Frame(container)
        inner_frame.pack()
        
        # Cargar y mostrar imagen
        try:
            if os.path.exists(image_path):
                # Usar PIL para cargar y redimensionar la imagen
                original_image = Image.open(image_path)
                # Redimensionar a un tamaño apropiado
                new_size = (120, 120)  # ajusta según necesites
                resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
                btn_image = ImageTk.PhotoImage(resized_image)
                
                btn = ttk.Button(inner_frame, image=btn_image, command=command)
                btn.image = btn_image  # Mantener referencia
                btn.pack(side=tk.LEFT)
            else:
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        except Exception as e:
            print(f"No se pudo cargar la imagen {image_path}: {e}")
            # Fallback con emoji
            emoji = "" if "Etapas" in title else "⏱️"
            btn = ttk.Button(inner_frame, text=emoji, command=command, 
                           style='Large.TButton', width=20)
            btn.pack(side=tk.LEFT)
        
        # Botón de información
        info_btn = ttk.Button(inner_frame, text="ℹ️", style='Info.TButton', width=3)
        info_btn.pack(side=tk.RIGHT, padx=(10, 0))
        ToolTip(info_btn, tooltip_text)

class StagesSelectionFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de selección de etapas."""
        # Botón de volver
        ttk.Button(self, text="←", command=lambda: self.app._go_back(check_unsaved=False)).place(x=10, y=10)
        
        # Título
        tk.Label(self, text="Seleccione una etapa", 
                font=("Arial", 14, "bold")).pack(pady=15)
        
        # Mostrar imagen de etapas redimensionada
        self.show_etapas_image()
        
        # Frame para los botones
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=5, expand=True)
        
        # Botones de Etapas con información
        # Etapa 1: Entrada
        input_frame = ttk.Frame(buttons_frame)
        input_frame.pack(pady=8, fill=tk.X)
        
        ttk.Button(input_frame, text="1. Entrada de Señal x[n]", 
                  command=self.app.show_input_options, width=25).pack(side=tk.LEFT)
        
        info_input = ttk.Button(input_frame, text="ℹ️", style='Info.TButton', width=3)
        info_input.pack(side=tk.RIGHT, padx=10)
        ToolTip(info_input, "Carga o genera la señal de entrada\n\n"
                "Opciones:\n"
                "• Muestreo simulado: Recoge datos del ADC\n"
                "• Cargar archivo: WAV, MP3 o CSV\n"
                "• Grabación por micrófono:Captura audio\n en tiempo real")
        
        # Etapa 2: Filtro
        filter_frame = ttk.Frame(buttons_frame)
        filter_frame.pack(pady=8, fill=tk.X)
        
        ttk.Button(filter_frame, text="2. Diseño de Filtro H(z)", 
                  command=self.app.show_filter_options, width=25).pack(side=tk.LEFT)
        
        info_filter = ttk.Button(filter_frame, text="ℹ️", style='Info.TButton', width=3)
        info_filter.pack(side=tk.RIGHT, padx=10)
        ToolTip(info_filter, "Diseña filtros digitales IIR o FIR\n\n"
                "Modo Básico: Filtros simples:\nVentana Rectangular o Butterworth\n"
                "Modo Avanzado: Chebyshev, Elíptico,\nventanas FIR\n"
                "Tipos: Pasa-bajas, Pasa-altas, \nPasa-banda, Rechaza-banda")
        
        # Etapa 3: Salida
        output_frame = ttk.Frame(buttons_frame)
        output_frame.pack(pady=8, fill=tk.X)
        
        ttk.Button(output_frame, text="3. Salida y[n] = x[n] * h[n]", 
                  command=self.app.show_output_options, width=25).pack(side=tk.LEFT)
        
        info_output = ttk.Button(output_frame, text="ℹ️", style='Info.TButton', width=3)
        info_output.pack(side=tk.RIGHT, padx=10)
        ToolTip(info_output, "Aplica el filtro y compara resultados\n\n"
                "Características:\n"
                "• Aplicación de filtro a la señal de entrada\n"
                "• Visualización comparativa\n"
                "• Reproducción de señales original y filtrada\n"
                "• Exportación de resultados")

    def show_etapas_image(self):
        """Muestra la imagen de etapas redimensionada."""
        try:
            image_path = r"/home/pi/Desktop/etapas_icon.png"
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
            
            # Usar PIL para cargar y redimensionar la imagen
            original_image = Image.open(image_path)
            
            # Redimensionar a un tamaño pequeño y apropiado
            max_width = 250  # Ancho máximo
            max_height = 120  # Alto máximo
            
            # Calcular nuevas dimensiones manteniendo la proporción
            original_width, original_height = original_image.size
            ratio = min(max_width/original_width, max_height/original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convertir a PhotoImage de Tkinter
            etapas_image = ImageTk.PhotoImage(resized_image)
            
            # Crear label para la imagen
            image_label = tk.Label(self, image=etapas_image)
            image_label.image = etapas_image  # Mantener referencia
            image_label.pack(pady=5)
            
        except Exception as e:
            print(f"No se pudo cargar la imagen de etapas: {e}")
            # Crear un placeholder pequeño
            self.create_placeholder()

    def create_placeholder(self):
        """Crea un placeholder pequeño cuando no hay imagen."""
        placeholder_frame = tk.Frame(self, height=60, width=500, bg="#f0f0f0", relief="sunken")
        placeholder_frame.pack(pady=5)
        placeholder_frame.pack_propagate(False)
        
        placeholder_label = tk.Label(placeholder_frame, text=" Diagrama de Etapas", 
                                   font=("Arial", 9), bg="#f0f0f0", fg="#666666")
        placeholder_label.place(relx=0.5, rely=0.5, anchor="center")

class InputOptionsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de opciones de entrada."""
        # Botón de volver
        ttk.Button(self, text="←", command=lambda: self.app._go_back(check_unsaved=False)).place(x=10, y=10)
        
        tk.Label(self, text="Seleccione el origen de la señal", 
                font=("Arial", 14, "bold")).pack(pady=30)

        # Verificar qué archivos existen (para debug)
        self.check_existing_files()

        # Frame principal para los botones de imagen
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=20, expand=True)
        
        # Frame para los botones de imagen (uno al lado del otro)
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(pady=20)
        
        # Botones que BUSCAN imágenes primero, luego usan emoji como fallback
        buttons_data = [
            {
                "title": "Muestreo de Señal",
                "image_names": ["muestreo_icon.png", "muestreo_icon.jpg", "sampling_icon.png"],
                "fallback_emoji": "",
                "command": self.app.show_sampling_options,
                "tooltip": "Hace lectura de señal por medio SPI\n\nParámetros configurables:\n• Frecuencia de muestreo (Fs)\n• tiempo de muestreo (ts)\n• Duración de la señal"
            },
            {
                "title": "Cargar Archivos",
                "image_names": ["cargar_icon.png", "cargar_icon.jpg", "load_icon.png"],
                "fallback_emoji": "📁",
                "command": self.app.show_file_load_options,
                "tooltip": "Carga señales desde archivos\n\nFormatos soportados:\n• Audio: WAV, MP3\n• Datos: CSV (valores separados por:\n punto y coma (;))\nLa señal se normaliza y prepara para\n procesamiento"
            },
            {
                "title": "Grabación desde Micrófono",
                "image_names": ["grabacion_icon.png", "grabacion_icon.jpg", "record_icon.png"],
                "fallback_emoji": "🎤",
                "command": self.app.show_recording_options,
                "tooltip": "Graba audio desde el micrófono\n\nCaracterísticas:\n• Grabación en tiempo real\n• Control de grabación (pausa, reanudar, detener)\n• Visualización durante la grabación\n• Frecuencia de muestreo: 44100 Hz"
            }
        ]
        
        for button_data in buttons_data:
            self.create_image_button(images_frame, button_data)

    def check_existing_files(self):
        """Verifica qué archivos existen en el Desktop y directorio actual."""
        print("=== BUSCANDO ARCHIVOS DE IMAGEN ===")
        
        # Verificar Desktop
        desktop_path = "/home/pi/Desktop"
        if os.path.exists(desktop_path):
            print(f"📁 Contenido de {desktop_path}:")
            for file in os.listdir(desktop_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"     {file}")
        
        # Verificar directorio actual del script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📁 Contenido de {current_dir}:")
        for file in os.listdir(current_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"     {file}")

    def find_image(self, image_names):
        """Busca imágenes en múltiples ubicaciones."""
        search_paths = [
            "/home/pi/Desktop",                    # Desktop de Raspberry Pi
            os.path.dirname(os.path.abspath(__file__)),  # Directorio del script
            "/home/pi",                            # Home directory
        ]
        
        for path in search_paths:
            if not os.path.exists(path):
                continue
                
            for image_name in image_names:
                full_path = os.path.join(path, image_name)
                if os.path.exists(full_path):
                    print(f"🎯 IMAGEN ENCONTRADA: {full_path}")
                    return full_path
        
        print(f"   No se encontró: {image_names}")
        return None

    def create_image_button(self, parent, data):
        """Crea un botón que usa imagen si existe, sino emoji."""
        container = ttk.Frame(parent)
        container.pack(side=tk.LEFT, padx=25, pady=10)
        
        # Frame para el título y botón de info
        header_frame = ttk.Frame(container)
        header_frame.pack(pady=(0, 8))
        
        # Botón de información a la izquierda
        info_btn = ttk.Button(header_frame, text="ℹ️", style='Info.TButton', width=3)
        info_btn.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(info_btn, data["tooltip"])
        
        # Título
        title_label = tk.Label(header_frame, text=data["title"], 
                              font=("Arial", 11, "bold"), foreground="#003366")
        title_label.pack(side=tk.LEFT)
        
        # BUSCAR la imagen
        image_path = self.find_image(data["image_names"])
        
        if image_path:
            try:
                # Cargar y redimensionar imagen
                original_image = Image.open(image_path)
                new_size = (80, 80)  # Tamaño pequeño
                resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
                btn_image = ImageTk.PhotoImage(resized_image)
                
                # Crear botón con imagen
                btn = ttk.Button(container, image=btn_image, command=data["command"])
                btn.image = btn_image  # Mantener referencia
                btn.pack()
                print(f"  Botón con imagen creado: {data['title']}")
                return
                
            except Exception as e:
                print(f"   Error cargando imagen {image_path}: {e}")
        
        # FALLBACK: botón con emoji (si no se encuentra la imagen)
        print(f"🔄 Usando emoji para: {data['title']}")
        btn = ttk.Button(container, text=data["fallback_emoji"], 
                        command=data["command"], 
                        style='Large.TButton',
                        width=8)
        btn.pack()    

class SamplingOptionsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.is_sampling = False
        self.sampling_thread = None
        self.stop_sampling = threading.Event()
        self.spi = None
        self.setup_ui()
        self.initialize_spi()

        # Variables para reproducción (igual que recordingFrame)
        self.is_playing_signal = False
        self.is_paused_playback = False
        self.playback_position = 0
        self.playback_stream = None

    def initialize_spi(self):
        """Inicializa la comunicación SPI con el MCP3008"""
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # SPI0, CE0
            self.spi.max_speed_hz = 1000000  # 1 MHz
            self.spi.mode = 0  # Modo SPI 0
            print("SPI inicializado correctamente")
        except Exception as e:
            print(f"Error inicializando SPI: {e}")
            messagebox.showerror("Error SPI", f"No se pudo inicializar el SPI: {e}")

    def read_adc(self, channel):
        """Lee un canal específico del MCP3008 (0-7)"""
        if self.spi is None:
            return 0
            
        if channel < 0 or channel > 7:
            return 0 
        try:  
            # Comando para MCP3008
            adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
            data = ((adc[1] & 3) << 8) + adc[2]
            
            # Verificar que la lectura sea válida
            if data < 0 or data > 1023:
                print(f"ADVERTENCIA: Lectura ADC inválida: {data}")
                
            return data
        
        except Exception as e:
            print(f"Error leyendo ADC canal {channel}: {e}")
            return 0

    def adc_to_voltage(self, adc_value):
        """Convierte el valor ADC (0-1023) a voltaje (0-3.3V)"""
        return (adc_value * 3.3) / 1023.0

    def setup_ui(self):
        """Configura la interfaz para lectura SPI desde Raspberry Pi."""
        # Botón de volver
        ttk.Button(self, text="←", command=self.app._go_back).place(x=10, y=10)
        
        tk.Label(self, text="Lectura SPI - Raspberry Pi 4 con MCP3008", 
                font=("Arial", 14, "bold")).pack(pady=20)

        # Frame principal para controles
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Frame de información compacta
        info_frame = ttk.LabelFrame(main_frame, text="Información del Sistema", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        info_text = "ADC MCP3008 (10 bits, 0.0-3.0V) - Canal CH0\nFrecuencia: 100-9000 Hz  - Tiempo: 1-60 segundos"
        
        info_label = tk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                             font=("Arial", 10))
        info_label.pack(fill=tk.X)

        # Frame de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 20))

        # Frecuencia y tiempo en una sola línea
        params_frame = ttk.Frame(config_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(params_frame, text="Fs:").pack(side=tk.LEFT)
        self.freq_var = tk.StringVar(value="1000")
        ttk.Entry(params_frame, textvariable=self.freq_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(params_frame, text="Hz").pack(side=tk.LEFT)
        
        tk.Label(params_frame, text="ts:").pack(side=tk.LEFT)
        self.time_var = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.time_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(params_frame, text="seg").pack(side=tk.LEFT)

        # Canal ADC
        channel_frame = ttk.Frame(config_frame)
        channel_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(channel_frame, text="Canal ADC:").pack(side=tk.LEFT)
        self.channel_var = tk.StringVar(value="0")
        channel_combo = ttk.Combobox(channel_frame, textvariable=self.channel_var,
                                   values=["0"],
                                   state="readonly", width=5)
        channel_combo.pack(side=tk.LEFT, padx=5)

        # Botones de muestreo y acciones
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=10)

        # Botones de la izquierda: Muestrear, Mostrar Gráfica, Guardar Señal
        left_buttons_frame = ttk.Frame(actions_frame)
        left_buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.muestrear_btn = ttk.Button(left_buttons_frame, text="Muestrear", 
                                      command=self.start_spi_sampling, width=12)
        self.muestrear_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.mostrar_btn = ttk.Button(left_buttons_frame, text="Mostrar Gráfica", 
                                    command=self.show_graph, width=15, state=tk.DISABLED)
        self.mostrar_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.guardar_btn = ttk.Button(left_buttons_frame, text="Guardar Señal", 
                                    command=self.save_signal, width=12, state=tk.DISABLED)
        self.guardar_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Botones de reproducción a la derecha (al lado de Guardar Señal)
        right_buttons_frame = ttk.Frame(actions_frame)
        right_buttons_frame.pack(side=tk.RIGHT)

        # Frame para los botones de reproducción
        playback_frame = ttk.Frame(right_buttons_frame)
        playback_frame.pack(side=tk.RIGHT, padx=(10, 0))

        # Etiqueta "Reproducir:"
        ttk.Label(playback_frame, text="Reproducir:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 5))

        # Botones de reproducción compactos
        self.play_btn = ttk.Button(playback_frame, text="▶", 
                                 command=self.play_signal, width=3, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = ttk.Button(playback_frame, text="||", 
                                  command=self.pause_signal, width=3, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(playback_frame, text="■", 
                                 command=self.stop_signal, width=3, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        # Estado del sistema y lectura actual
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar(value="Sistema listo para muestrear")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, 
                                   font=("Arial", 10, "italic"), fg="blue")
        self.status_label.pack(side=tk.LEFT)

        self.current_reading_var = tk.StringVar(value="Lectura: -- V")
        reading_label = tk.Label(status_frame, textvariable=self.current_reading_var,
                                font=("Arial", 10), fg="green")
        reading_label.pack(side=tk.RIGHT)

        # Iniciar lectura en tiempo real del canal seleccionado
        self.update_current_reading()

    def update_current_reading(self):
        """Actualiza la lectura actual del canal seleccionado"""
        if not self.is_sampling and self.spi is not None:
            try:
                channel = int(self.channel_var.get())
                adc_value = self.read_adc(channel)
                voltage = self.adc_to_voltage(adc_value)
                self.current_reading_var.set(f"CH{channel}: {voltage:.3f} V")
            except Exception as e:
                self.current_reading_var.set(f"Error: {e}")
        
        # Programar próxima actualización
        self.after(500, self.update_current_reading)

    def start_spi_sampling(self):
        """Inicia el proceso de muestreo por SPI."""
        try:
            if self.spi is None:
                messagebox.showerror("Error SPI", "El SPI no está inicializado")
                return

            frecuencia = int(self.freq_var.get())
            tiempo = int(self.time_var.get())
            channel = int(self.channel_var.get())
            
            # Validar frecuencia (100-9000 Hz)
            if frecuencia < 100 or frecuencia > 9000:
                raise ValueError("Frecuencia debe estar entre 100 y 9000 Hz")
            if tiempo < 1 or tiempo > 60:
                raise ValueError("Tiempo debe estar entre 1 y 60 segundos")
            
            # ACTUALIZAR Fs CON LA FRECUENCIA DE MUESTREO REAL
            self.app.fs = frecuencia
            self.app._update_fs_display()  
            
            # Deshabilitar botones durante el muestreo
            self._set_sampling_buttons_state(False)
            
            # Cambiar a estado "muestreando" (amarillo)
            self.status_var.set(f"Muestreando CH{channel} a {frecuencia}Hz...")
            self.status_label.config(fg="orange")
            
            # Detener lectura en tiempo real durante el muestreo
            self.is_sampling = True
            self.stop_sampling.clear()
            
            # Iniciar muestreo en un hilo separado
            self.sampling_thread = threading.Thread(
                target=self._perform_real_spi_sampling, 
                args=(frecuencia, tiempo, channel),
                daemon=True
            )
            self.sampling_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")

    def _set_sampling_buttons_state(self, enabled):
        """Habilita o deshabilita los botones de muestreo"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.muestrear_btn.config(state=state)
        self.mostrar_btn.config(state=state)
        self.guardar_btn.config(state=state)

    def _set_playback_buttons_state(self, is_playing):
        """Actualiza el estado de los botones de reproducción"""
        if is_playing:
            self.play_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.play_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)

    def _perform_real_spi_sampling(self, freq, time_duration, channel):
        """Realiza el muestreo SPI real en un hilo separado."""
        try:
            total_samples = freq * time_duration
            samples = []
            timestamps = []
            
            start_time = time.time()
            sample_interval = 1.0 / freq
            
            print(f"Iniciando muestreo: {total_samples} muestras a {freq} Hz")
            
            for i in range(total_samples):
                if self.stop_sampling.is_set():
                    break
                
                # Leer del ADC
                adc_value = self.read_adc(channel)
                voltage = self.adc_to_voltage(adc_value)
                samples.append(voltage)
                timestamps.append(time.time() - start_time)
                
                # Actualizar progreso
                if i % 100 == 0:
                    progress = (i + 1) / total_samples * 100
                    self.app.root.after(0, lambda p=progress: self._update_progress(p))
                
                # Esperar para mantener la frecuencia
                next_sample_time = start_time + (i + 1) * sample_interval
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Crear objeto de señal
            signal_data = {
                'data': np.array(samples),
                'timestamps': np.array(timestamps),
                'fs': freq,
                'time': time_duration,
                'channel': channel,
                'voltage_range': (min(samples), max(samples)) if samples else (0, 0)
            }
            
            # Actualizar en el hilo principal
            self.app.root.after(0, lambda: self._on_sampling_complete(signal_data))
            
        except Exception as e:
            self.app.root.after(0, lambda: self._on_sampling_error(str(e)))

    def _update_progress(self, progress):
        """Actualiza el progreso del muestreo"""
        self.status_var.set(f"Muestreando... {progress:.1f}%")

    def _on_sampling_complete(self, signal_data):
        """Se llama cuando el muestreo se completa exitosamente."""
        # Guardar la señal en input_signal y establecer signal_loaded
        self.app.input_signal = signal_data['data']
        self.app.fs = signal_data['fs']
        self.app.signal_loaded = True
        self.app.current_channel = signal_data['channel']
        self.is_sampling = False
        
        # Habilitar botones nuevamente
        self._set_sampling_buttons_state(True)
        self._set_playback_buttons_state(False)
        self.play_btn.config(state=tk.NORMAL)  # Habilitar play
        
        # Cambiar a estado "completado" (verde)
        self.status_var.set("Muestreo completado correctamente")
        self.status_label.config(fg="green")
        
        # Mostrar resumen
        channel = signal_data['channel']
        min_v, max_v = signal_data['voltage_range']
        messagebox.showinfo("Muestreo Completado", 
                        f"Canal CH{channel}\n"
                        f"Muestras: {len(signal_data['data'])}\n"
                        f"Rango: {min_v:.3f}V - {max_v:.3f}V\n"
                        f"Duración: {signal_data['time']}s")
        
        # Restaurar a estado "listo" después de 2 segundos
        self.after(2000, self._reset_status)
    
    def _on_sampling_error(self, error_msg):
        """Se llama cuando hay un error en el muestreo."""
        self.is_sampling = False
        
        # Habilitar botones nuevamente
        self._set_sampling_buttons_state(True)
        
        self.status_var.set(f"Error en muestreo: {error_msg}")
        self.status_label.config(fg="red")
        
        # Restaurar después de 3 segundos
        self.after(3000, self._reset_status)

    def _reset_status(self):
        """Restaura el estado a "listo" (azul)."""
        self.status_var.set("Sistema listo para muestrear")
        self.status_label.config(fg="blue")

    def stop_spi_sampling(self):
        """Detiene el muestreo SPI"""
        self.stop_sampling.set()
        self.is_sampling = False

    def show_graph(self):
        """Muestra la gráfica de la señal adquirida con límites 0-3.3V"""
        if hasattr(self.app, 'input_signal') and self.app.input_signal is not None:
            # Establecer que hay una señal cargada
            self.app.signal_loaded = True
            
            # Crear ventana personalizada para mostrar con límites 0-3.3V
            def create_custom_plot_window(window):
                # Frame principal
                main_frame = ttk.Frame(window)
                main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Título con información del canal
                channel = getattr(self.app, 'current_channel', '0')
                title_frame = ttk.Frame(main_frame)
                title_frame.pack(fill=tk.X, pady=(0, 10))
                
                title_label = tk.Label(title_frame, 
                                    text=f"Señal Muestreada - CH{channel} (ADC MCP3008)", 
                                    font=("Arial", 12, "bold"))
                title_label.pack()
                
                info_label = tk.Label(title_frame,
                                    text=f"Fs: {self.app.fs} Hz | Muestras: {len(self.app.input_signal)} | Rango: 0-3.3V",
                                    font=("Arial", 10))
                info_label.pack()
                
                # Crear figura
                fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))
                fig.subplots_adjust(hspace=0.4)
                
                canvas = FigureCanvasTkAgg(fig, master=main_frame)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Barra de herramientas
                toolbar = NavigationToolbar2Tk(canvas, main_frame)
                toolbar.update()
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                
                # Graficar señal en dominio de tiempo con límites 0-3.3V
                time_vec = np.arange(len(self.app.input_signal)) / self.app.fs
                
                ax_time.plot(time_vec, self.app.input_signal, 'b-', linewidth=1.5)
                ax_time.set_title("Dominio de Tiempo", fontweight='bold', fontsize=10)
                ax_time.set_xlabel("Tiempo (s)", fontsize=11)
                ax_time.set_ylabel("Voltaje (V)", fontsize=11)
                ax_time.grid(True, alpha=0.3, linestyle='--')
                ax_time.set_ylim(0, 3.3)  # LÍMITE 0-3.3V
                ax_time.set_xlim(0, time_vec[-1])
                
                # Graficar espectro de frecuencia
                freq_pos, magnitude = self.app.compute_magnitude_spectrum(
                    self.app.input_signal, self.app.fs, db_scale=False
                )
                
                if freq_pos is not None and magnitude is not None:
                    ax_freq.plot(freq_pos, magnitude, 'r-', linewidth=1.5)
                    ax_freq.set_title("Espectro de Frecuencia", fontweight='bold', fontsize=10)
                    ax_freq.set_xlabel("Frecuencia (Hz)", fontsize=11)
                    ax_freq.set_ylabel("Magnitud", fontsize=11)
                    ax_freq.grid(True, alpha=0.3, linestyle='--')
                    ax_freq.set_xlim(0, self.app.fs/2)
                
                # Botón de guardar
                def save_plot():
                    filename = filedialog.asksaveasfilename(
                        defaultextension=".png",
                        filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
                        title="Guardar gráfica de muestreo"
                    )
                    if filename:
                        fig.savefig(filename, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Éxito", f"Gráfica guardada en {filename}")
                
                save_btn = ttk.Button(main_frame, text="Guardar Gráfica", command=save_plot)
                save_btn.pack(pady=10)
                
                canvas.draw()
            
            self.app._show_window(f"Gráfica: Señal SPI - CH{getattr(self.app, 'current_channel', '0')}", 
                                create_custom_plot_window)
        else:
            messagebox.showwarning("Sin datos", "Primero debe realizar un muestreo")
            
    def save_signal(self):
        """Guarda la señal actual en un archivo."""
        if hasattr(self.app, 'input_signal') and self.app.input_signal is not None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                try:
                    import pandas as pd
                    time_vec = np.arange(len(self.app.input_signal)) / self.app.fs
                    df = pd.DataFrame({
                        'tiempo_s': time_vec,
                        'voltaje_V': self.app.input_signal
                    })
                    df.to_csv(filename, index=False)
                    self.status_var.set(f"Señal guardada en {filename}")
                    self.status_label.config(fg="green")
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo guardar: {e}")
        else:
            messagebox.showwarning("Sin datos", "No hay señal para guardar")

    def play_signal(self):
        """Reproduce la señal muestreada como audio"""
        if self.app.input_signal is None:
            return
        
        if self.is_playing_signal and self.is_paused_playback:
            # Reanudar reproducción pausada
            self.is_paused_playback = False
            self.pause_btn.config(text="||")
            return
        
        # Detener reproducción actual si hay una
        self.stop_signal()
        
        # Iniciar nueva reproducción
        self.is_playing_signal = True
        self.is_paused_playback = False
        self.playback_position = 0
        
        try:
            # Convertir señal de voltaje a audio (normalizar a rango -1 a 1)
            signal_audio = self.app.input_signal.copy()
            signal_audio = (signal_audio - 1.65) / 1.65  # Centrar en 0 y escalar
            signal_audio = np.clip(signal_audio, -0.99, 0.99)
            
            #  CORREGIDO: Verificar y ajustar la frecuencia de muestreo
            current_fs = self.app.fs
            
            # Frecuencias de muestreo comúnmente soportadas
            supported_rates = [8000, 11025, 16000, 22050, 44100, 48000, 96000]
            
            # Si la Fs actual no está soportada, usar la más cercana soportada
            if current_fs not in supported_rates:
                # Encontrar la frecuencia más cercana soportada
                closest_fs = min(supported_rates, key=lambda x: abs(x - current_fs))
                print(f"⚠️  Fs {current_fs} Hz no soportada. Usando {closest_fs} Hz en su lugar.")
                playback_fs = closest_fs
                
                # Resamplear la señal si es necesario
                if len(signal_audio) > 1:
                    from scipy import signal as scipy_signal
                    num_samples = int(len(signal_audio) * playback_fs / current_fs)
                    signal_audio = scipy_signal.resample(signal_audio, num_samples)
            else:
                playback_fs = current_fs
            
            print(f"🔊 Reproduciendo a {playback_fs} Hz")
            
            def callback(outdata, frames, time, status):
                if self.is_paused_playback:
                    outdata.fill(0)
                    return
                
                start_pos = self.playback_position
                end_pos = start_pos + frames
                
                if start_pos >= len(signal_audio):
                    raise sd.CallbackStop
                
                chunk = signal_audio[start_pos:end_pos]
                
                if len(chunk) < frames:
                    outdata[:len(chunk)] = chunk.reshape(-1, 1)
                    outdata[len(chunk):].fill(0)
                    raise sd.CallbackStop
                else:
                    outdata[:] = chunk.reshape(-1, 1)
                
                self.playback_position = end_pos
            
            # Usar la frecuencia de muestreo ajustada
            self.playback_stream = sd.OutputStream(
                samplerate=playback_fs,  # Usar la Fs ajustada
                channels=1, 
                callback=callback,
                finished_callback=self._on_playback_finished
            )
            self.playback_stream.start()
            
            # Actualizar estado de botones
            self._set_playback_buttons_state(True)
            self.status_var.set(f"Reproduciendo a {playback_fs} Hz...")
            self.status_label.config(fg="green")
            
        except Exception as e:
            error_msg = str(e)
            if "-9997" in error_msg:
                # Error específico de frecuencia no soportada
                supported_rates = [8000, 11025, 16000, 22050, 44100, 48000]
                messagebox.showerror(
                    "Error de Frecuencia", 
                    f"Frecuencia de muestreo {self.app.fs} Hz no soportada.\n\n"
                    f"Frecuencias soportadas: {', '.join(map(str, supported_rates))} Hz\n\n"
                    f"Por favor, use una frecuencia de muestreo compatible."
                )
            else:
                messagebox.showerror("Error", f"No se pudo reproducir la señal: {e}")
            
            self.stop_signal()
 
    def pause_signal(self):
        """Pausa o reanuda la reproducción"""
        if not self.is_playing_signal:
            return
        
        self.is_paused_playback = not self.is_paused_playback
        if self.is_paused_playback:
            self.pause_btn.config(text="▶")
            self.status_var.set("Reproducción pausada")
        else:
            self.pause_btn.config(text="||")
            self.status_var.set("Reproduciendo señal muestreada...")

    def stop_signal(self):
        """Detiene la reproducción sin congelar la interfaz"""
        if not self.playback_stream:
            return

        # Cambiar estado inmediatamente en la interfaz
        self.status_var.set("Deteniendo reproducción...")
        self.status_label.config(fg="orange")
        self.stop_btn.config(state=tk.DISABLED)

        def stop_thread():
            try:
                stream = self.playback_stream
                if stream:
                    if stream.active:
                        stream.stop(ignore_errors=True)
                    time.sleep(0.05)
                    stream.close()
                    self.playback_stream = None

                sd.stop()
                time.sleep(0.05)

                # Actualiza la interfaz desde el hilo principal
                self.after(0, self._finish_stop_signal)

            except Exception as e:
                print(f"Error al detener reproducción: {e}")
                self.after(0, lambda: self.status_var.set(f"Error al detener reproducción: {e}"))

        threading.Thread(target=stop_thread, daemon=True).start()

    def _finish_stop_signal(self):
        """Actualiza la interfaz después de detener correctamente"""
        self.is_playing_signal = False
        self.is_paused_playback = False
        self.playback_position = 0

        self.status_var.set("Reproducción detenida.")
        self.status_label.config(fg="blue")
        self._set_playback_buttons_state(False)
        self.play_btn.config(state=tk.NORMAL)

    def _on_playback_finished(self):
        """Callback cuando la reproducción termina naturalmente"""
        self.after(0, self._finish_stop_signal)

    def __del__(self):
        """Cierra el SPI al destruir el frame"""
        if self.spi is not None:
            self.spi.close() 
            
class FileLoadOptionsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de carga de archivo."""
        # Botón de volver
        ttk.Button(self, text="←", command=self.app._go_back).place(x=10, y=10)
        
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=(50, 10), padx=20, fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(main_frame, text="Esperando selección de archivo...", 
                               font=("Arial", 20), fg="blue")
        self.status_label.pack(pady=(0, 10))
        
        # Frame para información de formatos
        format_frame = ttk.LabelFrame(main_frame, text="Formatos Soportados", padding=10)
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        format_text = tk.Text(format_frame, height=4, width=60, font=("Arial", 12))
        format_text.pack(fill=tk.BOTH)
        format_text.insert("1.0", 
            "• Archivos de audio: WAV, MP3 (convertidos a mono)\n"
            "• Archivos de datos: CSV (valores separados por punto y coma, dos columnas: Tiempo;Señal)\n\n"
            "Los archivos se normalizan automáticamente para evitar clipping.")
        format_text.config(state=tk.DISABLED)
        
        # Botones de acción
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=10, fill=tk.X)
        
        load_btn = ttk.Button(action_frame, text="Buscar Archivo", 
                             style='Large.TButton', command=self.start_loading, width=15)
        load_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(load_btn, "Selecciona un archivo WAV, MP3 o CSV para cargar")
        
        ttk.Button(action_frame, text="Mostrar Gráfica", 
                  style='Large.TButton', command=lambda: self.app.show_signal_plot_window(self.app.input_signal, self.app.fs, "Cargada")).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Guardar en Memoria", 
                  style='Large.TButton', command=self.app._save_current_signal).pack(side=tk.LEFT, padx=5)
      
    def _update_ui_on_load(self, result):
        signal_data, fs = result
        if signal_data is not None:
            self.app.input_signal, self.app.fs = signal_data, fs
            self.app.signal_loaded = True  # ¡IMPORTANTE! Establecer que hay señal cargada
            self.app._update_fs_display()
            self.status_label.config(text=f"✓ Archivo cargado: {len(signal_data)} muestras a {fs} Hz", fg="green")
        else:
            self.status_label.config(text="✗ No se pudo cargar el archivo", fg="red")
            self.app.signal_loaded = False
            
    def _get_usb_devices(self):
        """
        Devuelve una lista de rutas de memorias USB montadas en /media/pi.
        Ejemplo de salida: ['/media/pi/MI_USB', '/media/pi/USB2']
        """
        usb_base = "/media/pi"
        usb_devices = []

        if os.path.exists(usb_base):
            for d in os.listdir(usb_base):
                full_path = os.path.join(usb_base, d)
                if os.path.isdir(full_path):
                    usb_devices.append(full_path)

        if usb_devices:
            print(f"💽 USB detectadas: {usb_devices}")
        else:
            print("  WARNING: No se detectaron memorias USB montadas en /media/pi")

        return usb_devices

    def _load_task(self):
        """Tarea de carga de archivo - ahora con soporte para leer desde USB"""
        try:
            # 🔍 Buscar memorias USB conectadas
            usb_devices = self._get_usb_devices()

            # Determinar carpeta inicial para el diálogo
            if usb_devices:
                if len(usb_devices) == 1:
                    # Solo una USB detectada
                    initial_dir = usb_devices[0]
                    print(f"📂 Usando USB detectada: {initial_dir}")
                else:
                    # Varias USB detectadas → pedir al usuario elegir una
                    usb_names = [os.path.basename(u) for u in usb_devices]
                    choice = simpledialog.askstring(
                        "Seleccionar USB",
                        "Se detectaron varias memorias USB:\n\n" +
                        "\n".join(usb_names) +
                        "\n\nEscribe el nombre exacto de la que deseas usar:"
                    )

                    if choice and choice in usb_names:
                        initial_dir = os.path.join("/media/pi", choice)
                        print(f"  USB seleccionada: {initial_dir}")
                    else:
                        initial_dir = self.app.signals_dir
                        print("  WARNING: No se seleccionó USB, usando carpeta por defecto.")
            else:
                # Si no hay USB, usar la carpeta por defecto
                initial_dir = self.app.signals_dir
                print(f"📁 Usando carpeta por defecto: {initial_dir}")

            # 🧭 Diálogo para seleccionar archivo
            fpath = filedialog.askopenfilename(
                title="Seleccionar archivo de señal",
                initialdir=initial_dir,
                filetypes=[
                    ("Archivos de audio", "*.wav *.mp3"),
                    ("Archivos CSV", "*.csv"),
                    ("Todos los archivos", "*.*")
                ]
            )

            if not fpath:
                return None, None

            # Actualizar estado
            self.app.root.after(0, lambda: self.status_label.config(text="Cargando archivo...", fg="blue"))

            # Cargar según el tipo de archivo
            if fpath.lower().endswith(('.wav', '.mp3')):
                signal_data, fs = librosa.load(fpath, sr=None, mono=True)
                self.app.fs = fs
                self.app._update_fs_display()
                return signal_data, fs

            elif fpath.lower().endswith('.csv'):
                return self._load_csv_file(fpath)

            else:
                messagebox.showerror("Error", "Formato de archivo no soportado")
                return None, None

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
            print(f"Error al cargar archivo: {traceback.format_exc()}")
            return None, None
        
    def _load_csv_file(self, fpath):
        """Carga archivo CSV - VERSIÓN MEJORADA Y CORREGIDA"""
        try:
            print(f" Cargando archivo CSV: {fpath}")
            
            # Leer todo el contenido primero para análisis
            with open(fpath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            if not lines:
                raise ValueError("Archivo CSV vacío")
            
            print(f"📝 Archivo tiene {len(lines)} líneas")
            print(f"📖 Primeras 3 líneas:")
            for i in range(min(3, len(lines))):
                print(f"   Línea {i+1}: {lines[i].strip()}")
            
            # DETECCIÓN AUTOMÁTICA DE DELIMITADOR
            delimiter = self._detect_delimiter(lines[0])
            print(f"🔍 Delimitador detectado: '{delimiter}'")
            
            # PROCESAR DATOS
            time_data = []
            signal_data = []
            line_count = 0
            valid_lines = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue  # Saltar líneas vacías o comentarios
                    
                parts = line.split(delimiter)
                if len(parts) < 2:
                    print(f"  WARNING: Línea {i+1} ignorada: no tiene suficientes columnas")
                    continue
                
                try:
                    # Intentar convertir a números
                    tiempo = float(parts[0].strip().replace(',', '.'))  # Soporte para decimales con coma
                    señal = float(parts[1].strip().replace(',', '.'))
                    
                    time_data.append(tiempo)
                    signal_data.append(señal)
                    valid_lines += 1
                    
                except ValueError as e:
                    # Si falla la conversión, podría ser encabezado
                    if i == 0:  # Primera línea probablemente es encabezado
                        print(f"📋 Encabezado detectado: {line}")
                        continue
                    else:
                        print(f"   Error en línea {i+1}: {e} - Contenido: {line}")
                        continue
                
                line_count += 1
            
            if valid_lines < 2:
                raise ValueError(f"Solo se encontraron {valid_lines} líneas válidas. Se necesitan al menos 2")
            
            print(f"  {valid_lines} líneas válidas procesadas de {len(lines)} totales")
            
            # CALCULAR FRECUENCIA DE MUESTREO
            fs = self._calculate_sampling_rate(time_data, valid_lines)
            
            # CONVERTIR A ARRAY numpy
            signal_array = np.array(signal_data)
            
            # NORMALIZAR SEÑAL (opcional, para evitar clipping)
            max_val = np.max(np.abs(signal_array))
            if max_val > 1.0:
                signal_array = signal_array / max_val * 0.8
                print(f"📏 Señal normalizada (max era {max_val:.3f})")
            
            # ACTUALIZAR APLICACIÓN
            self.app.fs = fs
            self.app._update_fs_display()
            
            print(f"🎯 CSV cargado exitosamente:")
            print(f"   • Muestras: {len(signal_array)}")
            print(f"   • Fs: {fs:.2f} Hz")
            print(f"   • Duración: {time_data[-1] - time_data[0]:.3f} segundos")
            print(f"   • Rango de señal: [{np.min(signal_array):.3f}, {np.max(signal_array):.3f}]")
            
            return signal_array, fs
            
        except Exception as e:
            error_msg = f"Error procesando CSV:\n{str(e)}"
            print(f"   {error_msg}")
            messagebox.showerror("Error CSV", error_msg)
            return None, None

    def _detect_delimiter(self, first_line):
        """Detecta automáticamente el delimitador del CSV"""
        delimiters = [';', ',', '\t', '|']  # Orden de prioridad
        
        for delim in delimiters:
            parts = first_line.split(delim)
            if len(parts) >= 2:
                # Verificar que las partes puedan ser números (o al menos la primera línea)
                try:
                    if len(parts) == 2:
                        float(parts[0].strip().replace(',', '.'))
                        float(parts[1].strip().replace(',', '.'))
                        return delim
                except:
                    continue
                    
        # Si no se detecta, usar ; por defecto
        return ';'

    def _calculate_sampling_rate(self, time_data, num_samples):
        """Calcula la frecuencia de muestreo a partir de los datos de tiempo"""
        if len(time_data) < 2:
            return 44100  # Valor por defecto
        
        # Calcular diferencias de tiempo
        time_diffs = []
        for i in range(1, len(time_data)):
            dt = time_data[i] - time_data[i-1]
            if dt > 0:  # Solo diferencias positivas
                time_diffs.append(dt)
        
        if not time_diffs:
            return 44100  # Valor por defecto
        
        # Usar la mediana para evitar outliers
        median_dt = np.median(time_diffs)
        fs = 1.0 / median_dt
        
        # Validar que Fs sea razonable
        if fs < 1 or fs > 1000000:
            print(f"  WARNING: Fs calculada ({fs:.2f} Hz) fuera de rango, usando promedio")
            avg_dt = np.mean(time_diffs)
            fs = 1.0 / avg_dt
            
            if fs < 1 or fs > 1000000:
                print(f"  WARNING: Fs aún fuera de rango, usando 44100 Hz por defecto")
                return 44100
        
        print(f"📈 Fs calculada: {fs:.2f} Hz (dt = {median_dt:.6f} s)")
        return fs
    
    def start_loading(self):
        self.app._run_in_thread(self._load_task, self._update_ui_on_load)

class RecordingOptionsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de grabación."""
        # Botón de volver
        ttk.Button(self, text="←", command=self.app._go_back).place(x=10, y=10)
        
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=(50, 10), padx=20, fill=tk.BOTH, expand=True)
        
        # Estado de la grabación - CORREGIDO: usar tk.Label
        self.status_label = tk.Label(main_frame, text="Listo para grabar.", font=("Arial", 10))
        self.status_label.pack(pady=(0, 10))
        
        # Controles de grabación
        recording_frame = ttk.LabelFrame(main_frame, text="Controles de Grabación", padding=10)
        recording_frame.pack(pady=10, fill=tk.X)
        
        buttons_frame = ttk.Frame(recording_frame)
        buttons_frame.pack(pady=5)
        
        # Botones de grabación
        self.record_btn = ttk.Button(buttons_frame, text="● Rec", command=self.start_recording, 
                                   style='Large.TButton', width=10)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(buttons_frame, text="|| Pausa", command=self.pause_recording, 
                                  style='Large.TButton', state=tk.DISABLED, width=10)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(buttons_frame, text="■ Detener", command=self.stop_recording, 
                                 style='Large.TButton', state=tk.DISABLED, width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Controles de reproducción de la grabación
        playback_frame = ttk.LabelFrame(main_frame, text="Reproducción de Grabación", padding=10)
        playback_frame.pack(pady=10, fill=tk.X)
        
        playback_controls = ttk.Frame(playback_frame)
        playback_controls.pack(pady=5)
        
        # Botones de reproducción para escuchar lo grabado
        self.play_btn = ttk.Button(playback_controls, text="▶ Play", command=self.play_recording,
                                  style='Large.TButton', state=tk.DISABLED, width=10)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_playback_btn = ttk.Button(playback_controls, text="|| Pausa", command=self.pause_playback,
                                           style='Large.TButton', state=tk.DISABLED, width=10)
        self.pause_playback_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_playback_btn = ttk.Button(playback_controls, text="■ Stop", command=self.stop_playback,
                                          style='Large.TButton', state=tk.DISABLED, width=10)
        self.stop_playback_btn.pack(side=tk.LEFT, padx=5)
        
        # Botones de acción
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(action_frame, text="Mostrar Gráfica", 
          command=lambda: self.app.show_signal_plot_window(self.app.input_signal, self.app.fs, "Grabada")).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Guardar en Memoria", 
                  command=self.app._save_current_signal).pack(side=tk.LEFT, padx=5)
        
        # Inicializar variables de grabación
        self.app.recorded_frames = []
        self.app.is_recording = False
        self.app.is_paused_recording = False
        self.app.recording_stream = None
        
        # Variables para reproducción
        self.app.is_playing_recording = False
        self.app.is_paused_playback = False
        self.app.playback_position = 0
        self.app.playback_stream = None
    
    def start_recording(self):
        """Inicia la grabación"""
        # ANTES de grabar, asegurar que Fs está configurada correctamente
        self.app.fs = 44100  # O la Fs que uses para grabación
        self.app._update_fs_display()  # ¡ACTUALIZAR!
        
        self.app.start_recording()
        if self.app.is_recording:
            self.record_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Grabando...", foreground="green")
            
    def pause_recording(self):
        """Pausa o reanuda la grabación"""
        if not self.app.is_recording:
            return
        
        self.app.is_paused_recording = not self.app.is_paused_recording
        if self.app.is_paused_recording:
            self.pause_btn.config(text="▶ Reanudar")
            # CORREGIDO: Usar foreground en lugar de fg para tk.Label
            self.status_label.config(text="Grabación pausada", foreground="orange")
        else:
            self.pause_btn.config(text="|| Pausa")
            # CORREGIDO: Usar foreground en lugar de fg para tk.Label
            self.status_label.config(text="Grabando...", foreground="green")
    
    def stop_recording(self):
        """Detiene la grabación"""
        try:
            print("Deteniendo grabación...")

            # Detiene el stream de grabación si existe
            if hasattr(self.app, 'stream') and self.app.stream:
                if self.app.stream.active:
                    self.app.stream.stop()
                time.sleep(0.05)  # 🔹 breve pausa para dejar que el hilo de PortAudio termine
                self.app.stream.close()
                self.app.stream = None  # 🔹 libera la referencia

            # Detiene la grabación de la app
            self.app.stop_recording()

            if hasattr(self.app, 'recorded_frames') and self.app.recorded_frames and \
            hasattr(self.app, 'input_signal') and self.app.input_signal is not None:
                duration = len(self.app.input_signal) / self.app.fs
                self.app.signal_loaded = True
                self.status_label.config(
                    text=f"Grabación finalizada ({duration:.2f}s)",
                    foreground="blue"
                )
                self.play_btn.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="Grabación cancelada", foreground="red")
                self.app.signal_loaded = False

        except Exception as e:
            traceback.print_exc()
            self.status_label.config(text=f"Error al detener grabación: {e}", foreground="red")

        # Restaurar estado de botones
        self.record_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="|| Pausa")
        self.stop_btn.config(state=tk.DISABLED)
    
    def play_recording(self):
        """Reproduce la grabación"""
        if self.app.input_signal is None:
            return
        
        if self.app.is_playing_recording and self.app.is_paused_playback:
            # Reanudar reproducción pausada
            self.app.is_paused_playback = False
            self.pause_playback_btn.config(text="|| Pausa")
            return
        
        # Detener reproducción actual si hay una
        self.stop_playback()
        
        # Iniciar nueva reproducción
        self.app.is_playing_recording = True
        self.app.is_paused_playback = False
        self.app.playback_position = 0
        
        try:
            def callback(outdata, frames, time, status):
                if self.app.is_paused_playback:
                    outdata.fill(0)
                    return
                
                start_pos = self.app.playback_position
                end_pos = start_pos + frames
                
                if start_pos >= len(self.app.input_signal):
                    raise sd.CallbackStop
                
                chunk = self.app.input_signal[start_pos:end_pos]
                
                if len(chunk) < frames:
                    outdata[:len(chunk)] = chunk.reshape(-1, 1)
                    outdata[len(chunk):].fill(0)
                    raise sd.CallbackStop
                else:
                    outdata[:] = chunk.reshape(-1, 1)
                
                self.app.playback_position = end_pos
            
            self.app.playback_stream = sd.OutputStream(
                samplerate=self.app.fs, 
                channels=1, 
                callback=callback,
                finished_callback=self.stop_playback
            )
            self.app.playback_stream.start()
            
            # Actualizar estado de botones
            self.play_btn.config(state=tk.DISABLED)
            self.pause_playback_btn.config(state=tk.NORMAL)
            self.stop_playback_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo reproducir: {e}")
            self.stop_playback()
    
    def pause_playback(self):
        """Pausa o reanuda la reproducción"""
        if not self.app.is_playing_recording:
            return
        
        self.app.is_paused_playback = not self.app.is_paused_playback
        if self.app.is_paused_playback:
            self.pause_playback_btn.config(text="▶ Reanudar")
        else:
            self.pause_playback_btn.config(text="|| Pausa")
    
    def stop_playback(self):
        """Detiene la reproducción sin congelar la interfaz"""
        if not self.app.playback_stream:
            return

        # Cambiar estado inmediatamente en la interfaz
        self.status_label.config(text="Deteniendo reproducción...", foreground="orange")
        self.stop_playback_btn.config(state=tk.DISABLED)

        def stop_thread():
            try:
                stream = self.app.playback_stream
                if stream:
                    # Evitar errores si el stream ya no está activo
                    if stream.active:
                        stream.stop(ignore_errors=True)
                    time.sleep(0.05)  # 🔹 espera a que PortAudio cierre hilos ALSA
                    stream.close()
                    self.app.playback_stream = None

                sd.stop()  # 🔹 asegura que ALSA libere la salida
                time.sleep(0.05)

                # Actualiza la interfaz desde el hilo principal
                self.after(0, self._finish_stop_playback)

            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda: self.status_label.config(
                    text=f"Error al detener reproducción: {e}", foreground="red"
                ))

        # Ejecutar en un hilo separado para no congelar la GUI
        threading.Thread(target=stop_thread, daemon=True).start()

    def _finish_stop_playback(self):
        """Actualiza la interfaz después de detener correctamente"""
        self.app.is_playing_recording = False
        self.app.is_paused_playback = False
        self.app.playback_position = 0

        self.status_label.config(text="Reproducción detenida.", foreground="blue")
        self.play_btn.config(state=tk.NORMAL)
        self.pause_playback_btn.config(state=tk.DISABLED, text="|| Pausa")
        self.stop_playback_btn.config(state=tk.DISABLED)

class FilterOptionsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de diseño de filtro."""
        # Botón de volver
        ttk.Button(self, text="←", command=lambda: self.app._go_back(check_unsaved=False)).place(x=10, y=10)
        
        top_controls = ttk.Frame(self)
        top_controls.pack(pady=(50, 10), padx=20, fill=tk.X)
        
        # --- Controles Superiores ---
        self.is_advanced = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_controls, text="Modo Avanzado", variable=self.is_advanced, 
                        command=self._toggle_filter_mode).pack(side=tk.RIGHT)
        ttk.Label(top_controls, textvariable=self.app.fs_display_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(top_controls, textvariable=self.app.nyquist_display_var).pack(side=tk.LEFT, padx=10)
        
        # --- Modo Básico ---
        self.basic_filter_frame = ttk.LabelFrame(self, text="Modo Básico", padding=10)
        self._setup_basic_filter_ui(self.basic_filter_frame)
        
        # --- Modo Avanzado ---
        self.advanced_filter_frame = ttk.LabelFrame(self, text="Modo Avanzado", padding=10)
        self._setup_advanced_filter_ui(self.advanced_filter_frame)
        
        # Mostrar modo básico por defecto
        self.basic_filter_frame.pack(pady=5, padx=20, fill=tk.X)
        
        # Frame para botones de acción
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Botones de acción con explicaciones
        design_btn = ttk.Button(button_frame, text="Diseñar Filtro", 
                               command=lambda: self.app._design_filter(self.is_advanced.get(), self))
        design_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Respuesta en Frecuencia", 
                  command=self.app.show_frequency_response_window).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Guardar Filtro", 
                  command=self.app.save_filter).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Cargar Filtro", command=self.app.load_filter).pack(side=tk.LEFT, padx=5)
        
    def _toggle_filter_mode(self):
        """Alterna entre modo básico y avanzado"""
        if self.is_advanced.get():
            self.basic_filter_frame.pack_forget()
            self.advanced_filter_frame.pack(pady=5, padx=20, fill=tk.X)
        else:
            self.advanced_filter_frame.pack_forget()
            self.basic_filter_frame.pack(pady=5, padx=20, fill=tk.X)
        
    def _setup_basic_filter_ui(self, frame):
        """Configura la interfaz de usuario para el modo básico"""
        # Tipo de filtro (FIR/IIR)
        ttk.Label(frame, text="Clase:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.bf_class = ttk.Combobox(frame, values=["IIR", "FIR"], width=8, state="readonly")
        self.bf_class.grid(row=0, column=1)
        self.bf_class.set("IIR")
        
        # Efecto del filtro con explicaciones
        ttk.Label(frame, text="Efecto:").grid(row=0, column=2, sticky=tk.W, padx=15, pady=2)
        self.bf_effect = ttk.Combobox(frame, values=["Pasa-Bajas", "Pasa-Altas", "Pasa-Banda", "Rechaza-Banda"], 
                                    width=15, state="readonly")
        self.bf_effect.grid(row=0, column=3)
        self.bf_effect.set("Pasa-Bajas")
        
        # Frecuencia de corte inferior - CAMBIAR NOMBRE
        ttk.Label(frame, text="F_L (Hz):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)  # CAMBIADO
        self.bf_fc = ttk.Entry(frame, width=8)
        self.bf_fc.grid(row=2, column=1)
        self.bf_fc.insert(0, "1000")
        
        # Frecuencia de corte superior - CAMBIAR NOMBRE      
        self.bf_bw_label = ttk.Label(frame, text="F_H (Hz):", state=tk.DISABLED)  # CAMBIADO
        self.bf_bw_label.grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.bf_bw = ttk.Entry(frame, width=8, state=tk.DISABLED)
        self.bf_bw.grid(row=2, column=3)
        self.bf_bw.insert(0, "2000")
        
        # Orden del filtro
        ttk.Label(frame, text="Orden:").grid(row=1, column=0, sticky=tk.W, padx=15, pady=2)
        self.bf_order = ttk.Spinbox(frame, from_=1, to=15, width=5)
        self.bf_order.grid(row=1, column=1)
        self.bf_order.set("8")
        
        # Explicaciones de efectos
        effect_explanations = {
            "Pasa-Bajas": "Permite frecuencias bajas, atenúa altas",
            "Pasa-Altas": "Permite frecuencias altas, atenúa bajas", 
            "Pasa-Banda": "Permite banda específica de frecuencias",
            "Rechaza-Banda": "Rechaza banda específica de frecuencias"
        }
        
        self.bf_effect_explanation = tk.Label(frame, text=effect_explanations["Pasa-Bajas"], 
                                             font=("Arial", 8), fg="blue", wraplength=200)
        self.bf_effect_explanation.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5)
                      
        # Configurar eventos para actualizar la UI dinámicamente
        self.bf_class.bind("<<ComboboxSelected>>", lambda e: self._update_basic_ui())
        self.bf_effect.bind("<<ComboboxSelected>>", lambda e: [self._update_basic_ui(), self._update_basic_ui_with_explanation()])
        
        # Actualizar UI inicialmente
        self._update_basic_ui()
        self._update_basic_ui_with_explanation()
    
    def _update_basic_ui(self, event=None):
        """Actualiza dinámicamente la UI del modo básico"""
        try:
            # Actualizar rango de orden según tipo de filtro
            if self.bf_class.get() == "FIR":
                self.bf_order.config(from_=100, to=10000)
                current_val = self.bf_order.get()
                if not current_val.isdigit() or int(current_val) < 100:
                    self.bf_order.set("100")
            else:  # IIR
                self.bf_order.config(from_=1, to=15)
                current_val = self.bf_order.get()
                if not current_val.isdigit() or int(current_val) > 15:
                    self.bf_order.set("4")
            
            # Habilitar/deshabilitar frecuencia superior según tipo de filtro
            is_band_type = self.bf_effect.get() in ["Pasa-Banda", "Rechaza-Banda"]
            bw_state = tk.NORMAL if is_band_type else tk.DISABLED
            self.bf_bw_label.config(state=bw_state)
            self.bf_bw.config(state=bw_state)
            
        except Exception as e:
            print(f"Error en actualización de UI básica: {e}")
    
    def _update_basic_ui_with_explanation(self, event=None):
        """Actualiza la explicación del efecto seleccionado en modo básico"""
        try:
            effect_explanations = {
                "Pasa-Bajas": "Permite frecuencias bajas, atenúa altas",
                "Pasa-Altas": "Permite frecuencias altas, atenúa bajas", 
                "Pasa-Banda": "Permite banda específica de frecuencias",
                "Rechaza-Banda": "Rechaza banda específica de frecuencias"
            }
            
            selected_effect = self.bf_effect.get()
            if selected_effect in effect_explanations:
                self.bf_effect_explanation.config(text=effect_explanations[selected_effect])
                
        except Exception as e:
            print(f"Error en actualización de explicaciones básicas: {e}")        
    
    def _setup_advanced_filter_ui(self, frame):
        """Configura la interfaz de usuario para el modo avanzado"""
        # Tipo de filtro (FIR/IIR)
        ttk.Label(frame, text="Tipo de filtro:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.af_class = ttk.Combobox(frame, values=["IIR", "FIR"], width=8, state="readonly")
        self.af_class.grid(row=0, column=1)
        self.af_class.set("IIR")
        
        # Configuración del filtro con explicaciones
        ttk.Label(frame, text="Configuración:").grid(row=0, column=2, sticky=tk.W, padx=15, pady=2)
        
        # Configuraciones diferentes para IIR y FIR
        self.af_config = ttk.Combobox(frame, width=20, state="readonly")
        self.af_config.grid(row=0, column=3)
        
        # Efecto del filtro con explicaciones 
        
        ttk.Label(frame, text="Efecto:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.af_effect = ttk.Combobox(frame, values=["Pasa-Bajas", "Pasa-Altas", "Pasa-Banda", "Rechaza-Banda"], 
                                    width=15, state="readonly")
        self.af_effect.grid(row=1, column=1)
        self.af_effect.set("Pasa-Bajas")
        
        # Frecuencia de corte inferior - NOMBRE CONSISTENTE
        ttk.Label(frame, text="F_L (Hz):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)  # YA ESTÁ CORRECTO
        self.af_fc_low = ttk.Entry(frame, width=8)
        self.af_fc_low.grid(row=3, column=1)
        self.af_fc_low.insert(0, "1000")
        
        # Explicaciones de configuraciones
        config_explanations = {
            "Butterworth": "Respuesta plana en banda de paso",
            "Chebyshev I": "Ripple en banda de paso, transición más rápida",
            "Chebyshev II": "Ripple en banda de rechazo", 
            "Elíptica": "Ripple en ambas bandas, transición más rápida",
            "Ventana Kaiser": "FIR con ventana Kaiser (óptimo)",
            "Ventana Hamming": "FIR con ventana Hamming",
            "Ventana Blackman": "FIR con ventana Blackman",
            "Ventana Rectangular": "FIR con ventana rectangular"
        }
        
        self.af_config_explanation = tk.Label(frame, text=config_explanations["Butterworth"], 
                                         font=("Arial", 8), fg="blue", wraplength=200)
        self.af_config_explanation.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Explicaciones de efectos
        effect_explanations = {
            "Pasa-Bajas": "Permite frecuencias bajas, atenúa altas",
            "Pasa-Altas": "Permite frecuencias altas, atenúa bajas", 
            "Pasa-Banda": "Permite banda específica de frecuencias",
            "Rechaza-Banda": "Rechaza banda específica de frecuencias"
        }
        
        self.af_effect_explanation = tk.Label(frame, text=effect_explanations["Pasa-Bajas"], 
                                            font=("Arial", 8), fg="blue", wraplength=200)
        self.af_effect_explanation.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # Orden del filtro
        ttk.Label(frame, text="Orden:").grid(row=2, column=2, sticky=tk.W, padx=15, pady=2)
        self.af_order = ttk.Spinbox(frame, from_=1, to=15, width=5)
        self.af_order.grid(row=2, column=3)
        self.af_order.set("4")
        
        # Frecuencia de corte superior (solo para tipos de banda)
        self.af_fc_high_label = ttk.Label(frame, text="F_H (Hz):", state=tk.DISABLED)
        self.af_fc_high_label.grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        self.af_fc_high = ttk.Entry(frame, width=8, state=tk.DISABLED)
        self.af_fc_high.grid(row=3, column=3)
        self.af_fc_high.insert(0, "2000")
        
        # Ripple en banda de paso (solo para Chebyshev I y Elíptica)
        self.af_ripple_label = ttk.Label(frame, text="Ripple Banda Paso (dB):", state=tk.DISABLED)
        self.af_ripple_label.grid(row=4, column=0, sticky=tk.W, padx=15, pady=2)
        self.af_ripple = ttk.Entry(frame, width=8, state=tk.DISABLED)
        self.af_ripple.grid(row=4, column=1)
        self.af_ripple.insert(0, "1.0")
        
        # Ripple en banda de rechazo (solo para Chebyshev II y Elíptica)
        self.af_stopband_label = ttk.Label(frame, text="Ripple Banda Rechazo (dB):", state=tk.DISABLED)
        self.af_stopband_label.grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)
        self.af_stopband = ttk.Entry(frame, width=8, state=tk.DISABLED)
        self.af_stopband.grid(row=4, column=3)
        self.af_stopband.insert(0, "40.0")
        
        # Configurar eventos para actualizar la UI dinámicamente
        self.af_class.bind("<<ComboboxSelected>>", self._on_af_class_change)
        self.af_config.bind("<<ComboboxSelected>>", lambda e: [self._update_advanced_ui(), self._update_advanced_ui_with_explanation()])
        self.af_effect.bind("<<ComboboxSelected>>", lambda e: [self._update_advanced_ui(), self._update_advanced_ui_with_explanation()])
        
        # Actualizar UI inicialmente
        self._update_advanced_config_options()
        self._update_advanced_ui()
        self._update_advanced_ui_with_explanation()
    
    def _update_advanced_ui(self, event=None):
        """Actualiza dinámicamente la UI del modo avanzado"""
        try:
            # Mostrar/ocultar controles según tipo de filtro
            is_iir = self.af_class.get() == "IIR"
            is_fir = not is_iir
            
            # Configurar controles de configuración
            config_state = tk.NORMAL
            self.af_config.config(state=config_state)
            
            # Configurar rango de orden
            if is_fir:
                self.af_order.config(from_=100, to=10000)
                current_val = self.af_order.get()
                if not current_val.isdigit() or int(current_val) < 100:
                    self.af_order.set("100")
            else:  # IIR
                self.af_order.config(from_=1, to=15)
                current_val = self.af_order.get()
                if not current_val.isdigit() or int(current_val) > 15:
                    self.af_order.set("4")
            
            # Habilitar/deshabilitar frecuencia superior según tipo de filtro
            is_band_type = self.af_effect.get() in ["Pasa-Banda", "Rechaza-Banda"]
            band_state = tk.NORMAL if is_band_type else tk.DISABLED
            self.af_fc_high_label.config(state=band_state)
            self.af_fc_high.config(state=band_state)
            
            # Configurar controles de ripple según tipo de filtro y configuración
            if is_iir:
                config_type = self.af_config.get()
                
                # Ripple en banda de paso (Chebyshev I y Elíptica)
                ripple_state = tk.NORMAL if config_type in ["Chebyshev I", "Elíptica"] else tk.DISABLED
                self.af_ripple_label.config(state=ripple_state)
                self.af_ripple.config(state=ripple_state)
                
                # Ripple en banda de rechazo (Chebyshev II y Elíptica)
                stopband_state = tk.NORMAL if config_type in ["Chebyshev II", "Elíptica"] else tk.DISABLED
                self.af_stopband_label.config(state=stopband_state)
                self.af_stopband.config(state=stopband_state)
            else:
                # Deshabilitar todos los controles de ripple para FIR
                self.af_ripple_label.config(state=tk.DISABLED)
                self.af_ripple.config(state=tk.DISABLED)
                self.af_stopband_label.config(state=tk.DISABLED)
                self.af_stopband.config(state=tk.DISABLED)
                
        except Exception as e:
            print(f"Error en actualización de UI avanzada: {e}")
    
    def _update_advanced_ui_with_explanation(self, event=None):
        """Actualiza las explicaciones en modo avanzado"""
        try:
            # Explicaciones de configuraciones
            config_explanations = {
                "Butterworth": "Respuesta plana en banda de paso",
                "Chebyshev I": "Ripple en banda de paso, transición más rápida",
                "Chebyshev II": "Ripple en banda de rechazo", 
                "Elíptica": "Ripple en ambas bandas, transición más rápida",
                "Ventana Kaiser": "FIR con ventana Kaiser (óptimo)",
                "Ventana Hamming": "FIR con ventana Hamming",
                "Ventana Blackman": "FIR con ventana Blackman",
                "Ventana Rectangular": "FIR con ventana rectangular"
            }
            
            # Explicaciones de efectos
            effect_explanations = {
                "Pasa-Bajas": "Permite frecuencias bajas, atenúa altas",
                "Pasa-Altas": "Permite frecuencias altas, atenúa bajas", 
                "Pasa-Banda": "Permite banda específica de frecuencias",
                "Rechaza-Banda": "Rechaza banda específica de frecuencias"
            }
            
            selected_config = self.af_config.get()
            if selected_config in config_explanations:
                self.af_config_explanation.config(text=config_explanations[selected_config])
            
            selected_effect = self.af_effect.get()
            if selected_effect in effect_explanations:
                self.af_effect_explanation.config(text=effect_explanations[selected_effect])
                
        except Exception as e:
            print(f"Error en actualización de explicaciones: {e}")
    
    def _on_af_class_change(self, event):
        """Maneja el cambio de tipo de filtro en modo avanzado"""
        self._update_advanced_config_options()
        self._update_advanced_ui()
        self._update_advanced_ui_with_explanation()
    
    def _update_advanced_config_options(self):
        """Actualiza las opciones de configuración según el tipo de filtro seleccionado"""
        if self.af_class.get() == "IIR":
            config_options = ["Butterworth", "Chebyshev I", "Chebyshev II", "Elíptica"]
            self.af_config.config(values=config_options)
            self.af_config.set("Butterworth")
        else:  # FIR
            config_options = ["Ventana Kaiser", "Ventana Hamming", "Ventana Blackman", "Ventana Rectangular"]
            self.af_config.config(values=config_options)
            self.af_config.set("Ventana Kaiser")

class OutputOptionsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.fig = None
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de salida SIMPLIFICADA"""
        # Botón de volver
        ttk.Button(self, text="← Volver", command=self.app._go_back).place(x=10, y=10)
        
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=(50, 5), padx=20, fill=tk.BOTH, expand=True)
        
        # Título
        title_label = tk.Label(main_frame, text="Salida y[n] = x[n] * h[n]", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Controles superiores
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Selección de señal y filtro
        ttk.Label(controls_frame, text="Señal:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.signal_var = tk.StringVar()
        signal_cb = ttk.Combobox(controls_frame, textvariable=self.signal_var, 
                               values=list(self.app.saved_signals.keys()), width=50, state="readonly")
        signal_cb.grid(row=0, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Filtro:", font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=5, sticky=tk.W)
        self.filter_var = tk.StringVar()
        filter_files = [f for f in os.listdir(self.app.filters_dir) if f.endswith('.json')]
        filter_cb = ttk.Combobox(controls_frame, textvariable=self.filter_var, 
                               values=filter_files, width=50, state="readonly")
        filter_cb.grid(row=1, column=1, padx=5)
        
        # Botón de aplicar filtro
        apply_btn = ttk.Button(controls_frame, text="Aplicar Filtro", 
                             command=self.apply_filter, width=15, style='Large.TButton')
        apply_btn.grid(row=0, column=2, padx=5)
        
        # Botón de guardar gráficas (SOLO cuando haya señal de salida)
        self.save_plots_btn = ttk.Button(controls_frame, text="Guardar Gráficas", 
                                       command=self.save_plots, 
                                       width=15, style='Large.TButton', state=tk.DISABLED)
        self.save_plots_btn.grid(row=1, column=2, padx=5)
                
        # Botones de acción
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=10, fill=tk.X)
        
        # Controles de reproducción 
        playback_frame = ttk.LabelFrame(main_frame, text="Controles de Reproducción", padding=10)
        playback_frame.pack(pady=10, fill=tk.X)
        
        playback_controls = ttk.Frame(playback_frame)
        playback_controls.pack(pady=5)
        
        # Entrada
        ttk.Label(playback_controls, text="Entrada:", font=('Arial', 10)).grid(row=0, column=0, padx=5)
        self.input_controls = self._create_enhanced_playback_controls(playback_controls, "input")
        self.input_controls.grid(row=0, column=1, padx=5)
        
        # Salida  
        ttk.Label(playback_controls, text="Salida:", font=('Arial', 10)).grid(row=0, column=2, padx=5)
        self.output_controls = self._create_enhanced_playback_controls(playback_controls, "output")
        self.output_controls.grid(row=0, column=3, padx=5)
        
        # Botones de exportación
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(pady=10, fill=tk.X)
        
        # Botón de guardar señal CSV
        ttk.Button(export_frame, text="Guardar CSV", 
          command=self._save_signals_csv, width=15, style='Large.TButton').pack(side=tk.LEFT, padx=5)
        
        # Botón de guardar señal WAV
        ttk.Button(export_frame, text="Guardar Señal Filtrada", 
                  command=self._save_filtered_signal_wav, 
                  width=20, style='Large.TButton').pack(side=tk.LEFT, padx=5)
        
        # Inicializar estado de botones usando nuestro propio sistema
        self._update_signal_availability_state("input", False)
        self._update_signal_availability_state("output", False)

    def _create_enhanced_playback_controls(self, parent, signal_type):
        """Crea controles de reproducción mejorados"""
        frame = ttk.Frame(parent)
        
        # Variables de estado para esta señal
        setattr(self, f'is_playing_{signal_type}', False)
        setattr(self, f'is_paused_{signal_type}', False)
        setattr(self, f'playback_position_{signal_type}', 0)
        setattr(self, f'playback_stream_{signal_type}', None)
        
        # Botones
        play_btn = ttk.Button(frame, text="▶", width=3,
                            command=lambda: self._play_signal(signal_type))
        play_btn.pack(side=tk.LEFT, padx=2)
        
        pause_btn = ttk.Button(frame, text="||", width=3, 
                              command=lambda: self._pause_signal(signal_type))
        pause_btn.pack(side=tk.LEFT, padx=2)
        
        stop_btn = ttk.Button(frame, text="■", width=3,
                            command=lambda: self._stop_signal(signal_type))
        stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Guardar referencias
        setattr(self, f'play_btn_{signal_type}', play_btn)
        setattr(self, f'pause_btn_{signal_type}', pause_btn)
        setattr(self, f'stop_btn_{signal_type}', stop_btn)
        
        return frame

    def _play_signal(self, signal_type):
        """Reproduce la señal especificada"""
        signal_data = getattr(self.app, f'{signal_type}_signal')
        if signal_data is None:
            return
        
        is_playing = getattr(self, f'is_playing_{signal_type}')
        is_paused = getattr(self, f'is_paused_{signal_type}')
        
        if is_playing and is_paused:
            # Reanudar reproducción pausada
            setattr(self, f'is_paused_{signal_type}', False)
            getattr(self, f'pause_btn_{signal_type}').config(text="||")
            return
        
        # Detener reproducción actual si hay una
        self._stop_signal(signal_type)
        
        # Iniciar nueva reproducción
        setattr(self, f'is_playing_{signal_type}', True)
        setattr(self, f'is_paused_{signal_type}', False)
        setattr(self, f'playback_position_{signal_type}', 0)
        
        try:
            def callback(outdata, frames, time, status):
                if getattr(self, f'is_paused_{signal_type}'):
                    outdata.fill(0)
                    return
                
                start_pos = getattr(self, f'playback_position_{signal_type}')
                end_pos = start_pos + frames
                
                if start_pos >= len(signal_data):
                    raise sd.CallbackStop
                
                chunk = signal_data[start_pos:end_pos]
                
                if len(chunk) < frames:
                    outdata[:len(chunk)] = chunk.reshape(-1, 1)
                    outdata[len(chunk):].fill(0)
                    raise sd.CallbackStop
                else:
                    outdata[:] = chunk.reshape(-1, 1)
                
                setattr(self, f'playback_position_{signal_type}', end_pos)
            
            stream = sd.OutputStream(
                samplerate=self.app.fs, 
                channels=1, 
                callback=callback,
                finished_callback=lambda: self._on_playback_finished(signal_type)
            )
            stream.start()
            
            setattr(self, f'playback_stream_{signal_type}', stream)
            
            # Actualizar estado de botones
            self._update_signal_buttons_state(signal_type, True)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo reproducir {signal_type}: {e}")
            self._stop_signal(signal_type)
    
    def _pause_signal(self, signal_type):
        """Pausa o reanuda la reproducción"""
        is_playing = getattr(self, f'is_playing_{signal_type}')
        if not is_playing:
            return
        
        is_paused = getattr(self, f'is_paused_{signal_type}')
        setattr(self, f'is_paused_{signal_type}', not is_paused)
        
        pause_btn = getattr(self, f'pause_btn_{signal_type}')
        if is_paused:
            pause_btn.config(text="||")
        else:
            pause_btn.config(text="▶")
    
    def _stop_signal(self, signal_type):
        """Detiene la reproducción de forma segura"""
        stream = getattr(self, f'playback_stream_{signal_type}')
        if not stream:
            return
        
        def stop_thread():
            try:
                if stream.active:
                    stream.stop(ignore_errors=True)
                time.sleep(0.05)
                stream.close()
                setattr(self, f'playback_stream_{signal_type}', None)
                
                sd.stop()
                time.sleep(0.05)
                
                self.after(0, lambda: self._finish_stop_signal(signal_type))
                
            except Exception as e:
                print(f"Error al detener {signal_type}: {e}")
                self.after(0, lambda: self._finish_stop_signal(signal_type))
        
        threading.Thread(target=stop_thread, daemon=True).start()
    
    def _finish_stop_signal(self, signal_type):
        """Actualiza la interfaz después de detener"""
        setattr(self, f'is_playing_{signal_type}', False)
        setattr(self, f'is_paused_{signal_type}', False)
        setattr(self, f'playback_position_{signal_type}', 0)
        
        self._update_signal_buttons_state(signal_type, False)
    
    def _on_playback_finished(self, signal_type):
        """Callback cuando la reproducción termina naturalmente"""
        self.after(0, lambda: self._finish_stop_signal(signal_type))
    
    def _update_signal_availability_state(self, signal_type, has_signal):
        """Actualiza la disponibilidad de los botones basado en si hay señal"""
        play_btn = getattr(self, f'play_btn_{signal_type}')
        pause_btn = getattr(self, f'pause_btn_{signal_type}')
        stop_btn = getattr(self, f'stop_btn_{signal_type}')
        
        if has_signal:
            play_btn.config(state=tk.NORMAL)
        else:
            play_btn.config(state=tk.DISABLED)
            pause_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.DISABLED)

    def _update_signal_buttons_state(self, signal_type, is_playing):
        """Actualiza el estado de los botones de reproducción durante la reproducción"""
        play_btn = getattr(self, f'play_btn_{signal_type}')
        pause_btn = getattr(self, f'pause_btn_{signal_type}')
        stop_btn = getattr(self, f'stop_btn_{signal_type}')
        
        if is_playing:
            play_btn.config(state=tk.DISABLED)
            pause_btn.config(state=tk.NORMAL)
            stop_btn.config(state=tk.NORMAL)
        else:
            # Solo habilitar play si hay señal disponible
            signal_data = getattr(self.app, f'{signal_type}_signal')
            if signal_data is not None:
                play_btn.config(state=tk.NORMAL)
            else:
                play_btn.config(state=tk.DISABLED)
            pause_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.DISABLED)

    def apply_filter(self):
        """Aplica el filtro con manejo de errores mejorado"""
        signal_name = self.signal_var.get()
        filter_name = self.filter_var.get()
        
        if not signal_name or not filter_name:
            messagebox.showwarning("Selección requerida", "Debes seleccionar una señal y un filtro.")
            return
        
        print(f"DEBUG apply_filter: Señal seleccionada: {signal_name}")
        print(f"DEBUG apply_filter: Filtro seleccionado: {filter_name}")
        
        # Mostrar indicador de carga
        self.status_label = ttk.Label(self, text="Aplicando filtro...", foreground="blue")
        self.status_label.place(x=400, y=15)
        self.update()
        
        def apply_task():
            try:
                # Cargar señal
                if signal_name not in self.app.saved_signals:
                    print(f"DEBUG: Señal '{signal_name}' no encontrada en saved_signals")
                    return "señal_no_encontrada"
                
                signal_data = self.app.saved_signals[signal_name]
                print(f"DEBUG: Señal cargada - tipo: {type(signal_data)}")
                
                if isinstance(signal_data, dict):
                    signal = signal_data.get('data')
                    fs = signal_data.get('fs', getattr(self.app, 'fs', 44100))
                    print(f"DEBUG: Señal es dict - data: {type(signal)}, fs: {fs}")
                else:
                    signal = signal_data
                    fs = getattr(self.app, 'fs', 44100)
                    print(f"DEBUG: Señal directa - tipo: {type(signal)}, fs: {fs}")
                
                # Cargar filtro
                filter_path = os.path.join(self.app.filters_dir, filter_name)
                print(f"DEBUG: Ruta del filtro: {filter_path}")
                
                if not os.path.exists(filter_path):
                    print(f"DEBUG: Archivo de filtro no existe: {filter_path}")
                    return "filtro_no_encontrado"
                
                with open(filter_path, 'r') as f:
                    filter_config = json.load(f)
                print(f"DEBUG: Filtro cargado - keys: {list(filter_config.keys())}")
                
                # Aplicar filtro
                filtered_signal = self._apply_filter_to_signal(signal, filter_config, fs)
                
                #   CORREGIDO: Verificar primero si es string (código de error)
                if isinstance(filtered_signal, str):
                    if filtered_signal == "filtro_inestable":
                        print("DEBUG: Filtro IIR inestable")
                        return "filtro_inestable"
                    elif filtered_signal == "error_nan":
                        print("DEBUG: Señal filtrada contiene NaN/Inf")
                        return "error_nan"
                    elif filtered_signal == "senal_cero":
                        print("DEBUG: Señal filtrada muy pequeña")
                        return "senal_cero"
                    else:
                        print(f"DEBUG: Código de error desconocido: {filtered_signal}")
                        return "error_filtrado"
                
                #   Si no es string, verificar si es None
                elif filtered_signal is None:
                    print("DEBUG: _apply_filter_to_signal retornó None")
                    return "error_filtrado"
                
                #   Si llegamos aquí, es un array numpy (éxito)
                else:
                    print("DEBUG: Filtrado completado exitosamente")
                    # Guardar resultados
                    self.app.input_signal = signal
                    self.app.output_signal = filtered_signal
                    self.app.fs = fs  # Asegurar que Fs se mantiene
                    return "éxito"
                        
            except Exception as e:
                print(f"DEBUG: Error en apply_task: {e}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                return str(e)
        
        def on_complete(result):
            self.status_label.destroy()
            print(f"DEBUG on_complete: Resultado: {result}")
            
            if result == "éxito":
                # Actualizar disponibilidad de señales
                self._update_signal_availability_state("input", True)
                self._update_signal_availability_state("output", True)
                self.save_plots_btn.config(state=tk.NORMAL)
                self.app.show_comparison_window()
                messagebox.showinfo("Éxito", "Filtro aplicado correctamente")
            elif result == "filtro_inestable":
                messagebox.showerror("Error", "El filtro IIR es inestable. Reduce el orden o usa filtro FIR.")
            elif result == "error_nan":
                messagebox.showerror("Error", "El filtro generó valores inválidos (NaN). Prueba con diferentes parámetros.")
            elif result == "senal_cero":
                messagebox.showwarning("Advertencia", "La señal filtrada es muy pequeña. El filtro puede estar atenuando demasiado.")
            else:
                messagebox.showerror("Error", f"No se pudo aplicar el filtro: {result}")
        
        # Ejecutar en hilo
        threading.Thread(target=lambda: self.app.root.after(0, 
            lambda: on_complete(apply_task())), daemon=True).start()
        
    def _apply_filter_to_signal(self, signal, filter_config, fs):
        """Aplica el filtro a la señal usando scipy.signal - CORREGIDO"""
        try:
            from scipy import signal as scipy_signal
            
            print(f"DEBUG: Iniciando filtrado - Señal: {type(signal)}, longitud: {len(signal)}")
            print(f"DEBUG: Configuración del filtro: {filter_config}")
            
            # Convertir a numpy array
            signal = np.array(signal, dtype=np.float64)
            print(f"DEBUG: Señal convertida a numpy - dtype: {signal.dtype}, shape: {signal.shape}")
            
            filter_type = filter_config.get('type', 'fir')
            
            # CORREGIDO: Los coeficientes están en el nivel superior, no en 'coefficients'
            b = filter_config.get('b', [])
            a = filter_config.get('a', [1.0])
            
            print(f"DEBUG: Tipo de filtro: {filter_type}")
            print(f"DEBUG: Coeficientes b: {b}, longitud: {len(b)}")
            print(f"DEBUG: Coeficientes a: {a}, longitud: {len(a)}")
            
            #   CORREGIDO: Usar if/elif correctamente
            if filter_type.lower() == 'fir':  # Usar lower() para case-insensitive
                if not b:
                    print("DEBUG: ERROR - No hay coeficientes b en filtro FIR")
                    return None
                
                print("DEBUG: Aplicando filtro FIR...")
                filtered = scipy_signal.lfilter(b, [1.0], signal)
                print(f"DEBUG: Filtro FIR aplicado - resultado: {type(filtered)}, longitud: {len(filtered)}")
                
            elif filter_type.lower() == 'iir':  # Usar lower() para case-insensitive
                if not b:
                    print("DEBUG: ERROR - No hay coeficientes b en filtro IIR")
                    return None
                
                # Verificar estabilidad
                poles = np.roots(a)
                print(f"DEBUG: Polos del filtro IIR: {poles}")
                print(f"DEBUG: Magnitud de los polos: {np.abs(poles)}")
                
                if np.any(np.abs(poles) >= 1.0):
                    print("DEBUG: ERROR - Filtro IIR inestable")
                    return "filtro_inestable"
                
                print("DEBUG: Aplicando filtro IIR...")
                filtered = scipy_signal.lfilter(b, a, signal)
                print(f"DEBUG: Filtro IIR aplicado - resultado: {type(filtered)}, longitud: {len(filtered)}")
            
            else:
                print(f"DEBUG: ERROR - Tipo de filtro desconocido: {filter_type}")
                return None
            
            # Asegurar formato correcto
            filtered = np.array(filtered, dtype=np.float64).flatten()
            print(f"DEBUG: Señal filtrada procesada - dtype: {filtered.dtype}, shape: {filtered.shape}")
            
            # Verificar resultados
            has_nan = np.any(np.isnan(filtered))
            has_inf = np.any(np.isinf(filtered))
            max_val = np.max(np.abs(filtered)) if len(filtered) > 0 else 0
            
            print(f"DEBUG: Verificación - NaN: {has_nan}, Inf: {has_inf}, Max: {max_val}")
            
            if has_nan or has_inf:
                print("DEBUG: ERROR - Señal contiene NaN o Inf")
                return "error_nan"
            
            if max_val < 1e-10:
                print("DEBUG: ADVERTENCIA - Señal muy pequeña")
                return "senal_cero"
            
            print(f"DEBUG: Filtrado exitoso - Rango: [{np.min(filtered):.6f}, {np.max(filtered):.6f}]")
            return filtered
                    
        except Exception as e:
            print(f"DEBUG: ERROR en _apply_filter_to_signal: {e}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return None
   
    def save_plots(self):
        """Guarda las gráficas de comparación"""
        if self.app.input_signal is None or self.app.output_signal is None:
            messagebox.showwarning("Sin datos", "No hay señales para graficar.")
            return
        
        # Crear figura temporal para la comparación
        fig, ((ax_in_time, ax_out_time), (ax_in_freq, ax_out_freq)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Comparación de Señales - Entrada vs Salida", fontsize=14, fontweight='bold')
        
        try:
            # Graficar señales de entrada y salida
            self._plot_comparison(ax_in_time, ax_out_time, ax_in_freq, ax_out_freq)
            
            # Guardar la figura
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Guardar gráficas de comparación"
            )
            
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Gráficas guardadas en {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron guardar las gráficas: {str(e)}")
        finally:
            plt.close(fig)  # Cerrar la figura para liberar memoria

    def _plot_comparison(self, ax_in_time, ax_out_time, ax_in_freq, ax_out_freq):
        """Grafica la comparación de forma RÁPIDA"""
        if self.app.input_signal is None or self.app.output_signal is None:
            return
        
        try:
            # 1. GRÁFICAS DE TIEMPO (MUESTREAR PARA MÁS VELOCIDAD)
            n_total = len(self.app.input_signal)
            # Usar solo 5000 puntos para mayor velocidad
            step = max(1, n_total // 5000)
            indices = slice(0, n_total, step)
            
            time_vec = np.arange(len(self.app.input_signal[indices])) / self.app.fs
            ax_in_time.plot(time_vec, self.app.input_signal[indices], 'b-', linewidth=0.5, alpha=0.7)
            ax_in_time.set_title("Entrada - Tiempo")
            ax_in_time.set_xlabel("Tiempo (s)")
            ax_in_time.set_ylabel("Amplitud")
            ax_in_time.grid(True, alpha=0.3)
            
            time_vec_out = np.arange(len(self.app.output_signal[indices])) / self.app.fs
            ax_out_time.plot(time_vec_out, self.app.output_signal[indices], 'r-', linewidth=0.5, alpha=0.7)
            ax_out_time.set_title("Salida - Tiempo")
            ax_out_time.set_xlabel("Tiempo (s)")
            ax_out_time.set_ylabel("Amplitud")
            ax_out_time.grid(True, alpha=0.3)
            
            # 2. GRÁFICAS DE FRECUENCIA (OPTIMIZADAS Y CORREGIDAS)
            def fast_spectrum(signal_data, fs):
                n = len(signal_data)
                n_fft = min(4096, n)  # FFT más pequeña para velocidad
                
                if n_fft < n:
                    start_idx = (n - n_fft) // 2
                    segment = signal_data[start_idx:start_idx + n_fft]
                else:
                    segment = signal_data
                
                # Sin ventana para máxima velocidad
                fft_result = np.fft.fft(segment)
                freq = np.fft.fftfreq(len(segment), 1/fs)
                
                ## Espectro unilateral 
                num_points = len(segment) // 2
                freq_pos = freq[:num_points]  # Frecuencias positivas (0 a Fs/2)
                magnitude = np.abs(fft_result[:num_points])
                
                # Normalización
                magnitude = magnitude / len(segment)
                if len(magnitude) > 1:
                    magnitude[1:] *= 2
                
                # Convertir a dB
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                return freq_pos, np.clip(magnitude_db, -120, 5)
            
            # Espectro de entrada
            freq_in, mag_in_db = fast_spectrum(self.app.input_signal, self.app.fs)
            ax_in_freq.plot(freq_in, mag_in_db, 'b-', linewidth=0.5, alpha=0.8)
            ax_in_freq.set_title("Entrada - Frecuencia")
            ax_in_freq.set_xlabel("Frecuencia (Hz)")
            ax_in_freq.set_ylabel("Magnitud (dB)")
            ax_in_freq.grid(True, alpha=0.3)
            ax_in_freq.set_ylim(-120, 5)
            ax_in_freq.set_xlim(0, self.app.fs/2)

            # Espectro de salida
            freq_out, mag_out_db = fast_spectrum(self.app.output_signal, self.app.fs)
            ax_out_freq.plot(freq_out, mag_out_db, 'r-', linewidth=0.5, alpha=0.8)
            ax_out_freq.set_title("Salida - Frecuencia")
            ax_out_freq.set_xlabel("Frecuencia (Hz)")
            ax_out_freq.set_ylabel("Magnitud (dB)")
            ax_out_freq.grid(True, alpha=0.3)
            ax_out_freq.set_ylim(-120, 5)
            ax_out_freq.set_xlim(0, self.app.fs/2)
            
            # Añadir línea de corte del filtro
            ax_in_freq.axvline(x=8000, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax_out_freq.axvline(x=8000, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error en gráficas: {e}")
            # Mostrar mensaje de error en las gráficas
            for ax in [ax_in_time, ax_out_time, ax_in_freq, ax_out_freq]:
                ax.clear()
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                    transform=ax.transAxes, ha='center', va='center')

    def _save_filtered_signal_wav(self):
        """Guarda la señal filtrada como archivo WAV - TODO EN ESTA CLASE"""
        if self.app.output_signal is None:
            messagebox.showwarning("Sin datos", "No hay señal filtrada para guardar.")
            return
        
        try:
            # Diagnóstico
            self._debug_signal_info(self.app.output_signal, "Señal filtrada")
            
            # Normalizar y convertir a formato WAV
            signal_data = self.app.output_signal
            
            # Asegurar que es numpy array
            if not isinstance(signal_data, np.ndarray):
                signal_data = np.array(signal_data, dtype=np.float64)
            
            # Aplanar si es multidimensional
            if signal_data.ndim > 1:
                signal_data = signal_data.flatten()
            
            # Normalizar a rango [-1, 1]
            max_val = np.max(np.abs(signal_data))
            if max_val > 0:
                signal_data = signal_data / max_val
            
            # Convertir a formato de 16 bits para WAV
            signal_int16 = np.int16(signal_data * 32767)
            
            # Verificar Fs
            fs = getattr(self.app, 'fs', 44100)
            if fs is None:
                fs = 44100
            
            # Diálogo para guardar
            filename = filedialog.asksaveasfilename(
                initialdir=getattr(self.app, 'signals_dir', '.'),
                title="Guardar señal filtrada",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                initialfile="salida_filtrada.wav"
            )
            
            if filename:
                # Intentar con scipy primero
                try:
                    from scipy.io import wavfile
                    wavfile.write(filename, fs, signal_int16)
                    messagebox.showinfo("Éxito", f"Señal guardada en {os.path.basename(filename)}")
                    
                except ImportError:
                    # Fallback con wave nativo
                    import wave
                    with wave.open(filename, 'w') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16 bits = 2 bytes
                        wav_file.setframerate(fs)
                        wav_file.writeframes(signal_int16.tobytes())
                    
                    messagebox.showinfo("Éxito", f"Señal guardada en {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar señal: {str(e)}")
            print(f"Error detallado: {traceback.format_exc()}")

    def _debug_signal_info(self, signal_data, name):
        """Muestra información de diagnóstico de la señal"""
        print(f"\n=== DIAGNÓSTICO Señal: {name} ===")
        print(f"Tipo: {type(signal_data)}")
        print(f"Longitud: {len(signal_data)}")
        
        if isinstance(signal_data, np.ndarray):
            print(f"Shape: {signal_data.shape}")
            print(f"Dtype: {signal_data.dtype}")
            print(f"Rango: [{np.min(signal_data):.6f}, {np.max(signal_data):.6f}]")
            print(f"¿Contiene NaN?: {np.any(np.isnan(signal_data))}")
            print(f"¿Contiene Inf?: {np.any(np.isinf(signal_data))}")
        else:
            signal_np = np.array(signal_data)
            print(f"Rango: [{np.min(signal_np):.6f}, {np.max(signal_np):.6f}]")
        
        print("=" * 40)

    def _save_signals_csv(self):
        """Guarda las señales de entrada y salida en un archivo CSV con Fs"""
        if self.app.input_signal is None or self.app.output_signal is None:
            messagebox.showwarning("Datos incompletos", "Se necesitan ambas señales (entrada y salida) para guardar.")
            return
        
        filename = filedialog.asksaveasfilename(
            initialdir=getattr(self.app, 'signals_dir', '.'),
            title="Guardar señales como CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="comparacion_señales.csv"
        )
        
        if filename:
            try:
                # Asegurar que ambas señales tengan la misma longitud
                min_length = min(len(self.app.input_signal), len(self.app.output_signal))
                input_data = self.app.input_signal[:min_length]
                output_data = self.app.output_signal[:min_length]
                
                # Obtener Fs
                fs = getattr(self.app, 'fs', 44100)
                
                # Crear datos para CSV con Fs
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Escribir encabezado
                    writer.writerow(['Tiempo', 'Entrada', 'Salida', 'Fs'])
                    
                    # Escribir datos
                    for i in range(min_length):
                        tiempo = i / fs  # Calcular tiempo en segundos
                        if i == 0:
                            writer.writerow([f"{tiempo:.6f}", input_data[i], output_data[i], fs])
                        else:
                            writer.writerow([f"{tiempo:.6f}", input_data[i], output_data[i], ''])
                
                messagebox.showinfo("Éxito", f"Señales guardadas en {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el archivo CSV: {str(e)}")
                print(f"Error al guardar CSV: {traceback.format_exc()}")

class RealtimeProcessingFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.spi = None
        self._after_id = None

        self.initialize_spi()
        self.setup_ui()

        # Control de timing
        self.last_sample_time = 0
        self.sample_count = 0
        self.actual_fs = 0

        # Buffers
        self.plot_buffer_in = deque(maxlen=1000)
        self.plot_buffer_out = deque(maxlen=1000)

        # Estados
        self.spi_reading = False
        self.spi_paused = False

    # -------------------- SPI --------------------
    def initialize_spi(self):
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 1350000
            self.spi.mode = 0
            print("✅ SPI inicializado correctamente")
        except Exception as e:
            messagebox.showerror("Error SPI", f"No se pudo inicializar SPI:\n{e}")
            self.spi = None

    def read_adc(self, channel=0):
        if not self.spi:
            return 0
        cmd = [1, (8 + channel) << 4, 0]
        adc = self.spi.xfer2(cmd)
        return ((adc[1] & 3) << 8) + adc[2]

    def adc_to_voltage(self, value):
        return (value * 3.3) / 1023.0

    # -------------------- UI --------------------
    def setup_ui(self):

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(main_frame, text="Procesamiento Tiempo Real",
                  font=("Arial", 14, "bold")).pack(pady=10)

        # Fs
        self.fs_var = tk.StringVar(value="1000")
        ttk.Label(main_frame, text="Fs (Hz):").pack()
        ttk.Entry(main_frame, textvariable=self.fs_var, width=10).pack()

        # Botones
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="▶ Start",
                                    command=self.start_spi_reading)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="■ Stop",
                                   command=self.stop_spi_reading,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Figura
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_ylim(0, 3.3)
        self.ax.set_xlim(0, 1000)
        self.ax.set_xlabel("Muestras")
        self.ax.set_ylabel("Voltaje")
        self.ax.grid(True)

        self.line_in, = self.ax.plot([], [], 'b-', linewidth=1, label="Entrada")
        self.line_out, = self.ax.plot([], [], 'r-', linewidth=1.5, label="Filtrada")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Iniciar loop de graficación independiente
        self.after(50, self._update_plot)  # 20 Hz

    # -------------------- Control SPI --------------------
    def start_spi_reading(self):

        if not self.spi:
            return

        try:
            fs = float(self.fs_var.get())
            if fs < 10:
                fs = 10
            if fs > 10000:
                fs = 10000
        except:
            fs = 1000

        self.target_interval = 1.0 / fs
        self.last_sample_time = time.perf_counter()
        self.sample_count = 0
        self.actual_fs = 0

        self.spi_reading = True
        self.spi_paused = False

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.plot_buffer_in.clear()
        self.plot_buffer_out.clear()

        self._update_spi_data()

    def stop_spi_reading(self):
        self.spi_reading = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # -------------------- Adquisición rápida --------------------
    def _update_spi_data(self):

        if not self.spi_reading:
            return

        now = time.perf_counter()

        if now - self.last_sample_time >= self.target_interval:

            adc_value = self.read_adc(0)
            vin = self.adc_to_voltage(adc_value)
            vout = self._apply_filter(vin)

            self.plot_buffer_in.append(vin)
            self.plot_buffer_out.append(vout)

            self.sample_count += 1

            if self.sample_count >= 100:
                elapsed = now - self.last_sample_time
                if elapsed > 0:
                    self.actual_fs = self.sample_count / elapsed
                self.sample_count = 0
                self.last_sample_time = now

        # Llamada lo más rápida posible
        self.after(0, self._update_spi_data)

    # -------------------- Filtro --------------------
    def _apply_filter(self, vin):

        if hasattr(self.app, "current_filter") and self.app.current_filter:

            b = self.app.current_filter['b']
            a = self.app.current_filter['a']

            if not hasattr(self, 'filter_state'):
                self.filter_state = signal.lfilter_zi(b, a)

            out, self.filter_state = signal.lfilter(b, a, [vin], zi=self.filter_state)
            return out[0]

        return vin

    # -------------------- Graficación lenta (20 Hz) --------------------
    def _update_plot(self):

        if not self.winfo_exists():
            return

        if len(self.plot_buffer_in) > 0:

            x_data = list(range(len(self.plot_buffer_in)))

            self.line_in.set_data(x_data, list(self.plot_buffer_in))
            self.line_out.set_data(x_data, list(self.plot_buffer_out))

            self.ax.set_xlim(0, len(x_data))
            self.ax.set_ylim(0, 3.3)

            self.ax.set_title(
                f"Tiempo Real - Fs real: {self.actual_fs:.1f} Hz"
            )

            self.canvas.draw_idle()

        self.after(50, self._update_plot)

# ------------------------------------------------------------------------------
# Punto de entrada de la aplicación
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SignalProcessorApp(root)
        root.mainloop()
    except Exception as e:
        # Fallback para mostrar errores graves en la consola
        print(f"ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        