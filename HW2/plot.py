import matplotlib.pyplot as plt

q1_table = ["1.23515         22.9548         60.977         66.549",
             "2.4703         29.8923        77.7802        85.3004",
             "4.9406         31.2033         89.973        98.4654",
             "9.8812         32.4783        97.5786        107.798",
            "19.7624         26.3943        101.751        111.576",
            "39.5248          29.669        104.514        111.165",
            "79.0496         32.6882        96.1842        114.612",
            "158.099         32.9524        97.2492        114.271",
            "316.198         33.6162        104.002        104.699"]

q2 = {'prob_size': ([32768, 65536, 131072, 262144, 524288, 1048576], 'Number of Nodes'),
      'bandwidth': ([9.11, 8.82, 8.07, 5.81, 4.91, 4.54], 'Device Bandwidth (GB/sec)')}

if __name__ == "__main__":
    q1 = {'char': ([], []),
          'uint': ([], []),
          'uint2': ([], [])}

    for row in q1_table:
        data = row.strip().split()
        assert(len(data) == 4)
        q1['char'][1].append(data[1])
        q1['uint'][1].append(data[2])
        q1['uint2'][1].append(data[3])
        for k in q1.keys():
            q1[k][0].append(data[0])

    plt.gca().set_color_cycle(['red', 'green', 'blue'])
    for k in sorted(q1.keys()):
        plt.plot(q1[k][0], q1[k][1], 'o-')
    plt.legend([k for k in sorted(q1.keys())])#, loc='upper left')

    plt.xlabel('Problem Size (MB)')
    plt.ylabel('Device Bandwidth (GB/sec)')
    plt.show()

    plt.plot(q2['prob_size'][0], q2['bandwidth'][0], 'ro-')
    plt.xlabel(q2['prob_size'][1])
    plt.ylabel(q2['bandwidth'][1])
    plt.show()

