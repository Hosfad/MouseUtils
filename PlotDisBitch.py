import math
import pickle
import os
import traceback

import numpy as np
import scipy as scipy


#hypothesis
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def circular_hist(ax, x, w, bins=16, density=False, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    print("BINS: ", bins)
    widths = np.diff(bins)
    print("WIDTHS: ", widths, np.sum(widths));

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = w
    # Otherwise plot frequency proportional to radius
    else:
        radius = w

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    # ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def do_hist(info, ax2=None, bins=None):
    print("median: " + str(np.median(info)) + " std: " + str(np.std(info)))
    if (bins is None):
        bins = (np.max(info) - np.min(info)) / (
                (2 * scipy.stats.iqr(info)) / (len(info) ** .3333))

    print("bins is " + str(bins))
    if (ax2 is None):
        plt.hist(info, int(bins))
    else:
        ax2.hist(info, int(bins))
    hist, edges = np.histogram(info, bins=int(bins))
    print("do_hist hist ")
    print(np.asarray(hist) / np.sum(hist))
    print("min: " + str(min(info)))
    print("max: " + str(max(info)))
    if (ax2 is None):
        plt.show()
    else:
        return ax2


def get_mouse_tuple(mouse_str):
    splitted = mouse_str.split(",")
    # print(splitted)
    return [float(splitted[0]), float(splitted[1]), float(splitted[2])]


def get_mouse_tuple_bot(mouse_str):
    splitted = mouse_str.split(",")
    return [float(splitted[1]), float(splitted[2]), float(splitted[0])]


path = "/home/hosfad/Desktop/DreamBotScripting/Firas's scripts/mouse.txt"

# for pkl in os.listdir("/home/userhome/Bots/"):
for pkl in [path]:

    try:
        millis = 50.0
        # print("on",pkl)
        if (not os.path.exists(pkl) or True):



            if(not os.path.exists(path)):
                # print("path no exist")
                continue
            mouse_file = open(path)
            data_str = mouse_file.read()
            data = data_str.split("\n")
            data = data[2:-1]
            # print("data",data)
            time = 0
            t = 0
            mouse_tuple = get_mouse_tuple;
            if (".com" in path):
                # print("BOT!!")
                mouse_tuple = get_mouse_tuple_bot
            pos = mouse_tuple(data[0])[0:2]
            avg = [0, 0, 0]

            import numpy as np

            # print("pos " + str(pos))
            times = []
            point_count = 0;
            for point in data:
                point_count+=1
                # if(point_count%100==0):
                #     print(point_count)
                # print("in loop")
                if ("True" in point or "False" in point):
                    continue;
                if (not "," in point):
                    continue
                tuple = np.asarray(mouse_tuple(point))
                if(tuple[2]==0):
                    continue
                if (t + tuple[2] >= millis):
                    weight = millis - t
                    tuple[2] -= millis - t
                    t += weight

                    avg += (weight / millis) * tuple
                    # print("result1: " + str(avg) + ", " + str(t))
                    avg[2] = millis
                    times.append(avg)
                    t = 0
                    avg = [0, 0, 0]
                else:
                    weight = tuple[2]
                    t += weight
                    avg += (weight / millis) * tuple

                if (tuple[2] > millis):
                    # print("result2: " + str(tuple))
                    times.append(tuple)
                    avg = [0, 0, 0]
                    t = 0
            # print("out of loop")
            paths=0
            # times = np.asarray(times)
            i = 0;
            while i < len(times)-1:
                data = times[i]
                next = times[i+1]
                if data[0]==next[0] and data[1]==next[1]:
                    times[i][2]+=times[i+1][2]
                    del times[i+1]
                else:
                    i += 1
            times = np.asarray(times)
            for data in times:
                if (data[2] >= 150):  # millis * ):
                    paths+=1
            if(paths>0):
                f= open(pkl+".raw","w")
                for time1 in times:
                    f.write(str(time1[0])+" " + str(time1[1])+" "+str(time1[2])+"\n")
                pickle.dump(times, open(pkl, "wb"))
            else:
                if(os.path.exists(pkl)):
                    os.remove(pkl)
                    os.remove(pkl+".raw")
    except:
        print("SOMETHING FUCKY")
        traceback.print_exc()
        continue

        # print(times)
    if( not os.path.exists(pkl)):
        continue
    mouse_data = pickle.load(open(pkl, "rb"))
    # print(mouse_data)

    paths = []
    current_path = []
    idle_time = [];
    idle_time_norm = [];
    active_time=[]
    active=0
    for data in mouse_data:
        if (data[2] >= 150):  # millis * ):
            # if(1000<=data[2]<1000*60):
            if(active>0):
                active_time.append(active)
            active=0;
            # print(data[2])
            idle_time_norm.append(data[2])
            idle_time.append(np.log(data[2]))
            if (len(current_path) > 0):
                paths.append(current_path)
                current_path = []
            continue
        else:
            active+=data[2]
            current_path.append(data[0:2])
    #4096
    idle_index = -1;
    active_index=-1
    signal=[]
    # while idle_index<len(idle_time_norm)-1 and active_index<len(active_time)-1:
    #     idle_index+=1
    #     active_index+=1
    #     while idle_time_norm[idle_index]>50:
    #         idle_time_norm[idle_index]-=50
    #         signal.append(-1)
    #     while active_time[active_index] > 50:
    #         active_time[active_index] -= 50
    #         signal.append(1)
    # print("sig",signal)
    # Number of sample points
    # print("sig smaller ",signal[8000:8000+2**14]);
    index=0
    # yf = signal[index:index+2**14]
    # yf2 =signal[index+2**14:index+2**14+2**14]
    window=1000
    yf=active_time[window:window+window]
    yf2=active_time[0:window]
    bins=np.histogram(np.hstack((yf,yf2)), bins=40)[1]
    # xf = fftfreq(N, T)[:N//2]
    import matplotlib.pyplot as plt
    # plt.hist(yf,bins)
    # plt.hist(yf2,bins)
    # plt.grid()
    # plt.show()
    efficiencies = []
    distances = []
    distances_bird = []
    speeds = []
    time_length = []
    angles = []
    intra_angles =[]
    percentages = []
    immediate_speed = []
    x_pos = []
    y_pos = []
    intra_speeds=[]
    fitts_x=[]
    fitts_y=[]
    # efficiency vs path time
    import scipy.interpolate as interp
    for path_index in range(len(paths)):
        # try:
            path = paths[path_index]

            # if (len(intra_speeds) > 1100):
            #     break;

            if (len(path) < 5):
                continue;
            # print("first path: " + str(paths[1]))
            path = path[0:-2]
            x, y = np.split(path, [-1], axis=1)
            dist = np.linalg.norm(path[0] - path[-1])
            # if(dist<400 or  dist > 550):
            #     continue;

            dist_count = 0

            def getAngle(a, b, c):
                ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
                return ang + 360 if ang < 0 else ang
            start = path[0]
            # for i in range(len(path)):
            #     path[i][0]-=start[0]
            #     path[i][1]-=start[1]
            #     path[i][0]/=100
            #     path[i][1]/=100
            dist = np.linalg.norm(path[0] - path[-1])
            # print("dist traveled :" + str(dist), path[0],path[-1])
            prev_point = path[0];
            current_point = path[0]

            end = path[-1]
            angle = (math.atan2(end[1] - current_point[1], end[0] - current_point[0])) / math.pi * 180 + 180
            percentThrough = 0
            intra_speed = []
            intra_angle = []
            for point in path[1:]:
                x_pos.append(point[0])
                y_pos.append(point[1])
                local_dist = np.linalg.norm(current_point - point)
                if(local_dist>0):
                    intra_speed.append(local_dist)
                temp=getAngle(prev_point,current_point,point);
                if(temp>180):
                    temp-=180
                intra_angle.append(temp)
                dist_count += local_dist
                percentThrough += 1 / len(path);
                if local_dist < 500 and dist < 250:
                    percentages.append(percentThrough)
                    immediate_speed.append(local_dist)
                prev_point=current_point
                current_point = point;
            # print("total path dist: " + str(dist_count) + " time " +str (millis * len(path)))\
            efficiency = (dist + .00000001) / (dist_count + .00000001)
            # if(efficiency<.1 and "genoveva" in pkl):
            #     continue
            if (efficiency > 1.01):
                print("efficiency: " + str(efficiency))
                quit();

            if (len(intra_speed) > 1):
                if 0 in intra_speed:
                    print("critical error, speed 0")
                    quit();
                speed_interp = np.zeros(100)
                speed_np = np.asarray(intra_speed)
                speed_np = speed_np / (np.max(speed_np) + .0001)

                arr1_interp = interp.interp1d(np.arange(speed_np.size), speed_np)
                arr1_compress = arr1_interp(np.linspace(0, speed_np.size - 1, speed_interp.size))
                # print("speed_np", speed_np)
                # print("speed_interp", arr1_compress)
                # plt.plot(arr1_compress);
                # plt.show()
                temp = list(arr1_compress)
                # temp.append(efficiency)
                intra_speeds.append(temp)



            #######################
            angles_interp = np.zeros(100)
            angles_np = np.asarray(intra_angle)

            arr1_interp = interp.interp1d(np.arange(angles_np.size), angles_np)
            arr1_compress = arr1_interp(np.linspace(0, angles_np.size - 1, angles_interp.size))
            # plt.plot(arr1_compress/np.max(arr1_compress));
            # plt.show()

            intra_angles.append(arr1_compress)



            import matplotlib.pyplot as plt




            # print((dist)/(dist_count))
            # if (efficiency < .98 and efficiency != 1):

            speed = 1000 * dist / (millis * len(path))
            # if (speed < 8000 and  dist < 2000 and millis * len(path) < 6000):
            efficiencies.append(efficiency)
            time_length.append(millis * len(path))
            # print("millis times len",millis*len(path))
            if(millis*len(path)>4):
                fitts_y.append(millis * len(path))
                fitts_x.append([np.log(dist/13+1)/np.log(2)])
            distances.append(dist_count + .01)
            distances_bird.append(dist + .01)
            speeds.append(speed)
            angles.append(angle)
    if(len(fitts_x)>4):
        reg = LinearRegression().fit(fitts_x, fitts_y)
        print("score,", reg.score(fitts_x, fitts_y), reg.coef_[0], reg.intercept_,np.sum(idle_time_norm)+np.sum(active),pkl)
    # print("score,", r2_score(fitts_x, fitts_y))
    # print("coef", reg.coef_)
    # print("intercept", reg.intercept_)
        # else:
        #     angles.append(angle)
    # except:
    #     print("IT FUCKING DIED")
    #     pass

# print("LENGTH OF ANGLES " ,str(time_length))
    # print("x: " + str(x) + " y: "+str(y));
    # plt.plot(x,y)
    # plt.show()
from scipy import stats
import matplotlib.pyplot as plt
# c = np.vstack((np.asarray(fitts_x), )).T
# print(c)


plt.scatter(fitts_x,np.asarray(fitts_y))
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4);
ax5 = fig.add_subplot(3, 2, 5, projection="polar");
# ax6 = fig.add_subplot(3, 2, 6, projection="polar");
ax6 = fig.add_subplot(3, 2, 6);
x = speeds
y = time_length
print("min speed ", min(speeds))
print("max speed ", max(speeds))

print("min time ", min(time_length))
print("max time ", max(time_length))

# x = immediate_speed
# y = percentages
# x=x[2000:4000]
# # x=np.log(x);
# y=y[2000:4000]
print("len of imeediate " + str(len(x)))
idle_time.insert(0,idle_time[0]);
corr, _ = pearsonr(idle_time[1:], idle_time[:-1])
print('Pearsons correlation: %.3f' % corr)
ax1.scatter(idle_time[1:], idle_time[:-1],s=1)
ax1.set_title("idle_length[n] vs idle_length[n+1] ")

# efficiencies = np.divide(1.0,efficiencies)
print("len of eff ", len(efficiencies))
do_hist(efficiencies, ax2, bins=30)
ax2.set_title("efficiency histogram")
print("min eff: " + str(1.0 / np.min(efficiency)))
# do_hist(distances)
# do_hist(speeds)
# plt.hist2d(x,y, norm=colors.LogNorm())
ax3.scatter(x, y, s=1)
ax3.set_title("speeds vs time length histogram")
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

hist, xedges, yedges = np.histogram2d(x, y, bins=40)

np.set_printoptions(suppress=True, threshold=np.inf)
print("hist: " + str(np.asarray(hist) / np.sum(hist)))
# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = np.ones_like(zpos)
dz = hist.ravel()

# do_hist(idle_time, ax4, 61)
do_hist(np.log(distances_bird), ax4, 61)
ax4.set_title("idle histogram")



circle_bins=20.0
angle_speed_list = []
for i in range(int(circle_bins)):
    angle_speed_list.append([0])
for angle, speed in zip(angles, speeds):
    index = int(angle / (360. / circle_bins) - .0001)
    print(index);
    angle_speed_list[index].append(speed)
medians_angle = []
for angle_speed in angle_speed_list:
    if len(angle_speed)>0  and not np.isnan(np.median(angle_speed)):
        medians_angle.append(np.median(angle_speed))
    else:
        medians_angle.append(0)
        # 8775516138

# angle_efficiency_list = []
#
# for i in range(int(circle_bins)):
#     angle_efficiency_list.append([])
# for angle, efficiency in zip(angles, efficiencies):
#     index = int(angle / (360. / circle_bins) - .0001)
#     print(index)
#     angle_efficiency_list[index].append(efficiency)
# medians_efficiency = []
# for angle_efficiency in angle_efficiency_list:
#     medians_efficiency.append(np.std(angle_efficiency))  # 8775516138


x = angles
y = time_length

circular_hist(ax5, np.asarray(angles), medians_angle, bins=int(circle_bins))
ax5.set_title("\nangle vs median speed " + str(len(speeds)), y=.92)
print("INTRA SPEEDS: ", np.asarray(intra_speeds).shape)
intra_speeds=np.average(intra_speeds,axis=0)
# intra_angles=np.std(intra_angles,axis=0)
print("INTRA SPEEDS: ", np.asarray(intra_speeds).shape)
print("INTRA SPEEDS: ", np.asarray(intra_speeds))
# ax6.plot(intra_angles)
ax6.plot(intra_speeds)
# circular_hist(ax6, np.asarray(angles), medians_efficiency, bins=40)
ax6.set_title("average speed profile")
ax6.set_xlabel("% through path")
ax6.set_ylabel("relative speed to rest of path")
# hist, xedges, yedges = np.histogram2d(x, y, bins=80)

# ax6.scatter(x,y)
# ax6.set_title("distances vs efficiencies")
# ax6.set_xlabel("distance")
# ax6.set_ylabel("efficiency")

# ax6.hist2d(x_pos,y_pos,bins=10,norm=colors.LogNorm())
# ax6.hist(distances,bins=10)

# ax6.set_title("angles vs speed")
# ax6.set_xlabel("angle")
# ax6.set_ylabel("speed")
# ax6.scatter(angles,speeds)


medians_angle = medians_angle / (np.min(medians_angle)+.0001)
print("medians angle: " + str(medians_angle))
print("MEDIANS: ", medians_angle / (np.max(medians_angle)+.00001))
hist, xedges, yedges = np.histogram2d(angles, speeds, bins=80)
hist3, edges = np.histogramdd((speeds, time_length, angles), bins=40)
# hist3/=np.sum(hist3)

hist3 = str(hist3);
hist3 = hist3.replace("[", "{")
hist3 = hist3.replace("]", "}")
hist3 = hist3.replace(" ", ",")
while (",," in hist3):
    hist3 = hist3.replace(",,", ",")
hist3 = hist3.replace(",}", "}")
print(hist3)

print("final static double minSpeed =", np.min(speeds), ";")
print("final static double maxSpeed =", np.max(speeds), ";")

print("final static double minTime =", np.min(time_length), ";")
print("final static double maxTime =", np.max(time_length), ";")

print("final static double minAngle =", np.min(angles), ";")
print("final static double maxAngle =", np.max(angles), ";")

plt.show()


