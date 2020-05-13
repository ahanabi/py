'''
shape = cir.shape
c_y, c_x, depth = int(shape[0] / 2), int(shape[1] / 2), shape[2]
x1 = c_x + c_x * 0.8
src = img.copy()
freq_list = []
for i in range(0, len(lines)):
    x = (x1 - c_x) * np.cos(i * 3.14 / 180) + c_x
    y = (x1 - c_x) * np.sin(i * 3.14 / 180) + c_y
    temp = result.copy()
    cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=3)
    tt = img.copy()
    tt[temp[:, :, 2] == 255] = 255
    c = img[temp[:, :, 2] == 255]
    points = c[c == 0]
    freq_list.append((len(points), i))
    cv2.imshow('temp', temp)
    cv2.imshow('line3', tt)
    cv2.waitKey(1)
print('当前角度：', max(freq_list, key=lambda x: x[0]), '度')

for line in lines[0]:
    rho = line[0]
    theta = line[1]
    rtheta = theta * (180 / np.pi)
    print('θ1:', rtheta)
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
        pt1 = (int(rho / np.cos(theta)), 0)
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        a = int(
            int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
        b = int(result.shape[0] / 2)
        pt3 = (a, b)
        pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
        cv2.putText(result, 'theta={}'.format(int(rtheta)), pt4, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.line(result, pt3, pt4, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        pt1 = (0, int(rho / np.sin(theta)))
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
        a = int(
            int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
        b = int(result.shape[0] / 2)
        pt3 = (a, b)
        pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
        cv2.line(result, pt3, pt4, (0, 0, 255), 2, cv2.LINE_AA)


for line in lines[2]:
    rho = line[0]
    theta = line[1]
    rtheta = theta * (180 / np.pi)
    print('θ2:', - rtheta - 90)
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
        pt1 = (int(rho / np.cos(theta)), 0)
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        a = int(
            int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
        b = int(result.shape[0] / 2)
        pt3 = (a, b)
        pt4 = (int(int(int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)) + a) / 2),
               int(int(int(b + result.shape[0]) / 2)))
        cv2.putText(result, 'theta2={}'.format(int(- rtheta - 90)), pt4, cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 255), 1)
        cv2.line(result, pt3, pt4, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        pt1 = (0, int(rho / np.sin(theta)))
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
        a = int(
            int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
        b = int(result.shape[0] / 2)
        pt3 = (a, b)
        pt4 = (int(int(int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)) + a) / 2),
               int(int(int(b + result.shape[0]) / 2)))
        cv2.line(result, pt3, pt4, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('line', result)
        return result
'''

'''
for i in range(0, len(lines)):
    rho, theta = lines[i][0][0], lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(cir, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('liness', cir)
'''
