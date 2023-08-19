 # 
 # This file is part of the misc_snippets distribution (https://github.com/peads/misc_snippets).
 # Copyright (c) 2023 Patrick Eads..
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

import sys
import serial
from time import sleep
from serial import PARITY_NONE
from serial import PARITY_EVEN
from serial import PARITY_ODD
from serial import PARITY_SPACE
from serial import PARITY_MARK
from serial import STOPBITS_ONE
from serial import STOPBITS_TWO
from serial import EIGHTBITS
from serial import SEVENBITS

def nextParity(i):
    return {
        0: PARITY_NONE,
        1: PARITY_EVEN,
        2: PARITY_ODD,
        3: PARITY_MARK,
        4: PARITY_SPACE
    }[i]


def nextStopBit(i):
    return {
        0: STOPBITS_ONE,
        1: STOPBITS_TWO
    }[i]


def nextNumBits(i):
    return {
        0: EIGHTBITS,
        1: SEVENBITS
    }[i]

def main(port='/dev/serial0', baudrate=4800):
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 5):
                bits = nextNumBits(i)
                parity = nextParity(k)
                stopb = nextStopBit(j)     
                
                print('-----------------------------------------------')
                print('baudrate=%s, bytesize=%s, parity=%s, stopbits=%s' % (baudrate, bits, parity, stopb))
                print('-----------------------------------------------')

                ser = serial.Serial(port=port,
                                    baudrate=baudrate,
                                    bytesize=bits,
                                    stopbits=stopb,
                                    parity=parity)
                table = []
                for lines in range(0, 10):
                    received_data = ser.read()  # read serial port
                    sleep(0.03)
                    data_left = ser.inWaiting()  # check for remaining byte
                    received_data += ser.read(data_left)

                    table.append([','.join([str(ord(c)) for c in received_data]), received_data.decode('ascii', 'replace'), received_data.decode('latin1')])
                for line in table:
                    print(u'{: <60} {: <20} {: <20}'.format(*line))
    
if __name__ == '__main__': 
    args = sys.argv[1:]
    if (len(args) < 2):
        main()
    else:
        main(args[0], args[1])
