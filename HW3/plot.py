import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = {'Global':([], {'2':[], '4':[], '8':[]}),
            'Block':([], {'2':[], '4':[], '8':[]})}

    order = None
    with open('output.txt', 'r') as output:
        for line in output.readlines():
            if ' ' not in line:
                continue
            else:
                l = line.strip().split()
                length = len(l)
                if length == 2:
                    order = l[1]
                    if int(l[0]) not in data['Global'][0]:
                        data['Global'][0].append(int(l[0]))
                    if int(l[0]) not in data['Block'][0]:
                        data['Block'][0].append(int(l[0]))
                elif length == 3:
                    data[l[0]][1][order].append(float(l[2]))
                else:
                    raise ValueError("error.")

    # Plot Q3.1
    plt.gca().set_color_cycle(['red', 'green', 'blue'])
    for k in sorted(data.keys()):
        plt.plot(data[k][0], data[k][1]['4'], 'o-')
    plt.legend([k for k in sorted(data.keys())])

    plt.title('Bandwidth vs Grid Size (Different Algorithms)')
    plt.xlabel('Grid size')
    plt.ylabel('Bandwidth (GB/sec)')
    plt.show()

    # Plot Q3.2
    for k in sorted(data['Block'][1].keys()):
        plt.plot(data['Block'][0], data['Block'][1][k], 'o-')
    plt.legend(["order = " + str(k) for k in sorted(data['Block'][1].keys())])

    plt.title('Bandwidth vs Grid Size (Algorithm 2)')
    plt.xlabel('Grid size')
    plt.ylabel('Bandwidth (GB/sec)')
    plt.show()

