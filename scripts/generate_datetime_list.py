import sys
import datetime

if __name__ == '__main__':
    t1_str = sys.argv[1]
    t2_str = sys.argv[2]
    hours = int(sys.argv[3])

    fmt = '%Y-%m-%d-%H:%M:%S'
 
    t1 = datetime.datetime.strptime( t1_str, fmt )
    t2 = datetime.datetime.strptime( t2_str, fmt )
 
    t = t1
    while t < t2:
        print('%s' % t.strftime(fmt))
        t += datetime.timedelta(hours=hours)