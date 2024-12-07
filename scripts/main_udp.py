import multiprocessing as mp
from handlandmarks_show_plot import start_real_time_plot
from handlandmarks_with_realsense import handlandmarks_with_realsense
from tcp_server import start_tcp_server
from udp import start_udp

if __name__ == '__main__':
    queue_handpose = mp.Manager().Queue(maxsize=1)
    queue_points = mp.Manager().Queue(maxsize=1)
    queue_handpose_sub = mp.Manager().Queue(maxsize=1)
    queue_points_sub = mp.Manager().Queue(maxsize=1)
    # queue_handpose = mp.Queue(maxsize=1)
    # queue_points = mp.Queue(maxsize=1)

    data_process = mp.Process(target=handlandmarks_with_realsense, args=(queue_handpose, queue_points, queue_handpose_sub, queue_points_sub))
    data_process.start()

    plot_process = mp.Process(target=start_real_time_plot, args=(queue_handpose_sub, queue_points_sub))
    plot_process.start()

    # tcp_process = mp.Process(target=start_tcp_server, args=(queue_handpose, queue_points))
    # tcp_process.start()

    udp_process = mp.Process(target=start_udp, args=(queue_handpose, queue_points))
    udp_process.start()

    data_process.join()
    plot_process.join()
    # tcp_process.join()
    udp_process.join()