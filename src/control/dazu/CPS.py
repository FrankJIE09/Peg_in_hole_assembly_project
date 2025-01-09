#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import time
from tokenize import Double
import xmlrpc.client
import socket
import os
import struct
from enum import IntEnum
import numpy as np


# from yaml import compose_all

class RbtFSM(IntEnum):
    enCPSState_UnInitialize = 0
    enCPSState_Initialize = 1  # 1 初始化
    # 电箱状态机
    enCPSState_ElectricBoxDisconnect = 2  # 2 与电箱控制板断开
    enCPSState_ElectricBoxConnecting = 3  # 3 连接电箱控制板
    enCPSState_EmergencyStopHandling = 4  # 4 急停处理
    enCPSState_EmergencyStop = 5  # 5 急停
    enCPSState_Blackouting48V = 6  # 6 正在切断本体供电
    enCPSState_Blackout48V = 7  # 7 本体供电已切断
    enCPSState_Electrifying48V = 8  # 8 正在准备给本体供电
    enCPSState_SafetyGuardErrorHandling = 9  # 9.安全光幕ErrorHanding
    enCPSState_SafetyGuardError = 10  # 10.安全光幕Error
    enCPSState_SafetyGuardHandling = 11  # 11 安全光幕处理
    enCPSState_SafetyGuard = 12  # 12 安全光幕
    # controller状态机
    enCPSState_ControllerDisconnecting = 13  # 13 正在反初始化控制器
    enCPSState_ControllerDisconnect = 14  # 14 控制器已处理于未初始化状态
    enCPSState_ControllerConnecting = 15  # 15 正在初始化控制器
    enCPSState_ControllerVersionError = 16  # 16 控制器版本过低错误
    enCPSState_EtherCATError = 17  # 17 EtherCAT错误
    enCPSState_ControllerChecking = 18  # 18 控制器初始化后检查状态
    # Robot状态机
    enCPSState_Reseting = 19  # 19 复位机器人
    enCPSState_RobotOutofSafeSpace = 20  # 20 机器人超出安全空间
    enCPSState_RobotCollisionStop = 21  # 21 机器人安全碰撞停车
    enCPSState_Error = 22  # 22 机器人错误
    enCPSState_Enabling = 23  # 23 机器人使能中
    enCPSState_Disable = 24  # 24 机器人去使能
    enCPSState_Moving = 25  # 25 机器人运动中
    enCPSState_LongJogMoving = 26  # 26 机器人长点动运动中
    enCPSState_RobotStopping = 27  # 27 机器人停止运动中
    enCPSState_Disabling = 28  # 28 机器人去使能中
    enCPSState_RobotOpeningFreeDriver = 29  # 29 机器人正在开启零力示教
    enCPSState_RobotClosingFreeDriver = 30  # 30 机器人正在关闭零力示教
    enCPSState_FreeDriver = 31  # 31 机器人处于零力示教
    enCPSState_RobotHolding = 32  # 32 机器人暂停
    enCPSState_Standy = 33  # 33 机器人就绪
    # script状态机
    enCPSState_ScriptRunning = 34  # 34 脚本运行中
    enCPSState_ScriptHoldHandling = 35  # 35 脚本暂停处理中
    enCPSState_ScriptHolding = 36  # 36 脚本暂停
    enCPSState_ScriptStopping = 37  # 37 脚本停止中
    enCPSState_ScriptStopped = 38  # 38 脚本已停止
    # HRApp
    enCPSState_HRAppDisconnected = 39  # 39 HRApp部件断开
    enCPSState_HRAppError = 40  # 40 HRApp部件错误
    # 负载辨识
    enCPSState_RobotLoadIdentify = 41  # 41 负载辨识
    # brake
    enCPSState_Braking = 42  # 42 开关抱闸中
    enCPSState_TemperatureTooLow = 43  # 43 温度过低
    # FT
    enCPSState_FTOpeningFreeDriver = 44  # 44 机器人正在开启力控零力示教
    enCPSState_FTClosingFreeDriver = 45  # 45 机器人正在关闭力控零力示教
    enCPSState_FTFreeDriver = 46  # 46 机器人处于力控零力示教


dic_ErrorCode = {
    39500: "跟机器人连接未建立",
    39501: "命令输入参数错误",
    39502: "命令响应中参数错误",
    39503: "Socket通讯错误(超时、接收异常等)",
    39504: "跟机器人连接错误"
}


class RbtClient(object):
    clientIP = '127.0.0.1'
    clientPort = 10003
    xmlrpcAddr = 'http://127.0.0.1:20000'
    m_bConnect = False

    # tcp = socket.socket()

    def __init__(self):
        return

    def Connect2CPS(self, hostName, nPort):
        try:
            self.xmlrpcAddr = 'http://'
            self.xmlrpcAddr += hostName
            self.xmlrpcAddr += ':20000'
            self.tcp = socket.socket()
            print(self.xmlrpcAddr)
            self.rpcClient = xmlrpc.client.ServerProxy(self.xmlrpcAddr)
            self.clientIP = hostName
            self.clientPort = nPort
            self.tcp.connect((self.clientIP, self.clientPort))
            self.m_bConnect = True
            return 0
        except:
            self.m_bConnect = False
            return

    def DisconnectFromCPS(self):
        self.tcp.close()
        return 0

    def sendHRLog(self, nLevel, msg):
        self.rpcClient.HRLog(int(nLevel), str(msg))

    def sendScriptFinish(self, errorCode):
        command = 'SendScriptFinish,0,' + str(errorCode) + ',;'
        self.tcp.send(command.encode())
        self.tcp.recv(self.clientPort).decode("utf-8", "ignore")

    def sendScriptError(self, msg):
        self.rpcClient.SendScriptError(str(msg), str(""))

    def isConnected(self):
        return self.m_bConnect

    def sendVarValue(self, boxID, rbtID, VarName, Value, result):
        if isinstance(Value, list, result):
            ValueStr = '['
            for i in range(0, 6):
                ValueStr += str(Value[i])
                if i != 5:
                    ValueStr += ','
            ValueStr += ']'
            # command = 'SendVarValue,'
            # for i in range(0,6):
            #    command += str(Value[i]) + ','
            # command += ';'
        else:
            ValueStr = str(Value)
        # gVar = globals()
        # gVar[VarName]=Value
        self.rpcClient.SendVarValue(str(VarName), ValueStr)
        # return retData

    def sendAndRecv(self, cmd, result):
        # print(cmd)
        try:
            self.tcp.send(cmd.encode())
            ret = self.tcp.recv(self.clientPort).decode("utf-8", "ignore")
            retData = ret.split(',')
            logmsg = '[script]sendAndRecv:' + cmd
            if len(retData) < 3:
                logmsg = logmsg + ' exit with ServerReturnError'
                self.sendHRLog(2, logmsg)
                os._exit(0)

            if retData[0] == "errorcmd":
                logmsg = logmsg + ' exit with errorcmd'
                self.sendHRLog(2, logmsg)
                os._exit(0)

            if retData[1] == "Fail":
                logmsg = logmsg + 'exit with Fail[' + retData[1] + ']'
                self.sendHRLog(2, logmsg)
                errorData = int(retData[2])
                return errorData

            del retData[0]
            del retData[0]
            retData.pop()
            result.clear()
            for i in range(0, len(retData)):
                result.append(retData[i])
        except:
            self.m_bConnect = False
            return 39500
        return 0


class CPSClient(object):
    clientIP = '127.0.0.1'
    clientPort = 10003
    xmlrpcAddr = 'http://127.0.0.1:20000'
    g_clients = []
    MaxBox = 5

    dic_FSM = {
        0: "未初始化",
        1: "初始化",
        0: "未初始化",
        1: "初始化",
        2: "与电箱控制板断开",
        3: "连接电箱控制板",
        4: "急停处理中",
        5: "急停",
        6: "正在切断本体供电",
        7: "本体供电已切断",
        8: "正在准备给本体供电",
        9: "安全光幕错误处理中",
        10: "安全光幕错误",
        11: "安全光幕处理中",
        12: "安全光幕",
        13: "正在反初始化控制器",
        14: "控制器已处理于未初始化状态",
        15: "正在初始化控制器",
        16: "控制器版本过低错误",
        17: "EtherCAT错误",
        18: "控制器初始化后检查",
        19: "正在复位机器人",
        20: "机器人超出安全空间",
        21: "机器人安全碰撞停车",
        22: "机器人错误",
        23: "机器人使能中",
        24: "机器人去使能",
        25: "机器人运动中",
        26: "机器人长点动运动中",
        27: "机器人停止运动中",
        28: "机器人去使能中",
        29: "机器人正在开启零力示教",
        30: "机器人正在关闭零力示教",
        31: "机器人处于零力示教",
        32: "机器人暂停",
        33: "机器人就绪",
        34: "脚本运行中",
        35: "脚本暂停处理中",
        36: "脚本暂停",
        37: "脚本停止中",
        38: "脚本已停止",
        39: "HRApp部件断开",
        40: "HRApp部件错误",
        41: "负载辨识",
        42: "开关抱闸中"
    }

    def __init__(self):
        for i in range(self.MaxBox):
            self.g_clients.append(RbtClient())
        return

    def _waitMotion(self, isblending):
        motionIndex = 0
        doneFlag = '0'
        movingFlag = '1'
        if isblending:
            motionIndex = 11
            doneFlag = '1'
            movingFlag = '0'

        time.sleep(0.02)
        nDisableCNT = 0
        while True:
            if (nDisableCNT >= 5):
                time.sleep(0.01)
                os._exit(0)
            ret = self.HRIF_ReadRobotState(ret)
            if (ret[1] == '0'):
                nDisableCNT += 1
                log = ('[script]EnableState[' + ret[2] + '],count[' + str(nDisableCNT) + '] error')
                continue
            else:
                nDisableCNT = 0

            if (ret[2] == '1' or ret[7] == '1' or ret[9] == '0' or ret[10] == '0'):
                log = ('[script]errorState[' + ret[2] + '],emergency[' + ret[7] + '],Electfify[' + ret[9] + ']')
                print(log)
                time.sleep(0.1)
                os._exit(0)
            elif ret[8] == '1':
                time.sleep(0.01)
                continue


            elif ret[6] == '1':
                time.sleep(0.01)
                continue


            elif ret[motionIndex] == doneFlag:
                log = ('[script]ret[' + str(motionIndex) + ']==' + ret[motionIndex])
                print(log)
                break


            elif ret[motionIndex] == movingFlag:
                log = ('ret[' + str(motionIndex) + ']==' + ret[motionIndex])
                time.sleep(0.01)
                continue

            else:
                log = ('[script]waitBlendingDone unknow status exit')
                print(log)
                os._exit(0)
        return

    #
    # part 1 初始化
    #

    '''
    *	@index : 1
    *	@param brief:连接机器人服务器
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_Connect(self, boxID, hostName, nPort):
        if boxID >= self.MaxBox:
            return 39501
        try:
            self.g_clients[boxID].Connect2CPS(hostName, nPort)
            return 0
        except:
            return 39504

    '''
    *	@index : 2
    *	@param brief:断开连接机器人服务器
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_DisConnect(self, boxID):
        if boxID >= self.MaxBox:
            return 39501
        try:
            self.g_clients[boxID].DisconnectFromCPS()
            return 0
        except:
            return 39504

    '''
    *	@index : 3
    *	@param brief:判断控制器是否连接
    *	@param boxID:电箱ID
    *	@param return: True：已连接，False：未连接
    '''

    def HRIF_IsConnected(self, boxID):

        return self.g_clients[boxID].isConnected()

    '''
    *	@index : 4
    *	@param brief:控制器断电(断开机器人供电，系统关机)
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_ShutdownRobot(self, boxID):
        result = []
        command = 'OSCmd,1,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:连接控制器电箱
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_Connect2Box(self, boxID):
        result = []
        command = 'ConnectToBox,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:机器人上电
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_Electrify(self, boxID):
        result = []
        command = 'Electrify,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:机器人断电
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_BlackOut(self, boxID):
        result = []
        command = 'BlackOut,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:连接控制器，连接过程中会启动主站，初始化从站，配置参数，检查配置，完成后跳转到去使能状态
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_Connect2Controller(self, boxID):
        result = []
        command = 'StartMaster,'
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:是否为模拟机器人
    *	@param boxID:电箱ID
    *	@param result[0]: 1:是模拟机器人 2:不是模拟机器人
    *	@param return: 错误码
    '''

    def HRIF_IsSimulateRobot(self, boxID, result):
        command = 'IsSimulation,'
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:控制器是否启动完成
    *	@param boxID:电箱ID
    *	@param result[0]: 1:已完成 2:未完成
    *	@param return: 错误码
    '''

    def HRIF_IsControllerStarted(self, boxID, result):
        command = 'ReadControllerState,'
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:读取控制器版本号
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]: 整体版本号
    *	@param result[1]: CPS 版本 
    *	@param result[2]: 控制器版本
    *	@param result[3]: 电箱版本 
    *	@param result[4]: id控制板固件版本
    *	@param result[5]: 控制板固件版本
    *	@param result[6]: 算法版本 
    *	@param result[7]: 固件版本 
    *	@param return: 错误码
    '''

    def HRIF_ReadVersion(self, boxID, rbtID, result):
        command = 'ReadVersion,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:读取机器人类型
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]: 机器人类型
    *	@param return:  错误码
    '''

    def HRIF_ReadRobotModel(self, boxID, rbtID, result):
        command = 'ReadRobotModel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 2 轴控制指令
    #

    '''
    *	@index : 1
    *	@param brief:机器人使能
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpEnable(self, boxID, rbtID):
        result = []
        command = 'GrpPowerOn,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:机器人去使能
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpDisable(self, boxID, rbtID):
        result = []
        command = 'GrpPowerOff,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:机器人复位
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpReset(self, boxID, rbtID):
        result = []
        command = 'GrpReset,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief: 停止命令
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpStop(self, boxID, rbtID):
        result = []
        command = 'GrpStop,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:暂停运动命令
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpInterrupt(self, boxID, rbtID):
        result = []
        command = 'GrpInterrupt,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:继续运动命令
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpContinue(self, boxID, rbtID):
        result = []
        command = 'GrpContinue,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:机器人关闭零力示教
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpCloseFreeDriver(self, boxID, rbtID):
        result = []
        command = 'GrpCloseFreeDriver,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:机器人打开零力示教
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_GrpOpenFreeDriver(self, boxID, rbtID):
        result = []
        command = 'GrpOpenFreeDriver,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 3 脚本控制指令
    #

    '''
    *	@index : 1
    *	@param brief:运行指定脚本函数
    *	@param boxID:电箱ID
    *	@param strFuncName : 指定脚本函数名称
    *	@param param : 参数
    *	@param result[0] : 函数名
    *	@param return: 错误码
    '''

    def HRIF_RunFunc(self, boxID, funcName, params, result):
        command = 'RunFunc,'
        command += funcName + ','
        for param in params:
            command += str(param) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:开始运行脚本
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_StartScript(self, boxID):
        result = []
        command = 'StartScript,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:停止运行脚本
    *	@param rbtID:机器人ID,一般为0
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_StopScript(self, boxID, rbtID):
        result = []
        command = 'StopScript,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:暂停运行脚本
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_PauseScript(self, boxID):
        result = []
        command = 'PauseScript,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:继续运行脚本
    *	@param boxID:电箱ID
    *	@param return: 错误码
    '''

    def HRIF_ContinueScript(self, boxID):
        result = []
        command = 'ContinueScript,;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 4 电箱控制指令
    #

    '''
    *	@index : 1
    *	@param brief:读取电箱信息
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 电箱连接状态
    *	@param result[1] : 48V电压状态
    *	@param result[2] : 48V输出电压值
    *	@param result[3] : 48V输出电流值
    *	@param result[4] : 远程急停状态
    *	@param result[5] : 三段按钮状态
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxInfo(self, boxID, rbtID, result):
        command = 'ReadBoxInfo,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:读取电箱控制数字输入状态
    *	@param boxID:电箱ID
    *	@param bit : 控制数字输入位
    *	@param result[0]: 数字输入状态
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxCI(self, boxID, bit, result):
        command = 'ReadBoxCI,' + str(bit) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:读取通用数字输入状态
    *	@param boxID:电箱ID
    *	@param bit : 通用数字输入位
    *	@param result[0]: 通用数字输入状态
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxDI(self, boxID, bit, result):
        command = 'ReadBoxDI,' + str(bit) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:读取控制器数字输出状态
    *	@param boxID:电箱ID
    *	@param bit : 控制器数字输出位
    *	@param result[0]: 数字输出状态
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxCO(self, boxID, bit, result):
        command = 'ReadBoxCO,' + str(bit) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:读取通用数字输出状态
    *	@param boxID:电箱ID
    *	@param bit : 通用数字输出位
    *	@param result[0]: 通用数字输出状态
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxDO(self, boxID, bit, result):
        command = 'ReadBoxDO,' + str(bit) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:读取模拟量输入值
    *	@param boxID:电箱ID
    *	@param bit : 模拟量输入位
    *	@param result[0] : 模拟量输入值
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxAI(self, boxID, bit, result):
        command = 'ReadBoxAI,' + str(bit) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:读取模拟量输出值
    *	@param boxID:电箱ID
    *	@param bit : 模拟量输出位
    *	@param result[0] : 模拟量输出状态 1:电压 2:电流
    *	@param result[1] : 模拟量输出值
    *	@param return: 错误码
    '''

    def HRIF_ReadBoxAO(self, boxID, bit, result):
        command = 'ReadBoxAO,' + str(bit) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:设置控制数字输出状态
    *	@param boxID:电箱ID
    *	@param bit : 控制数字输出位
    *	@param state : 设置的控股数字输出状态
    *	@param return: 错误码
    '''

    def HRIF_SetBoxCO(self, boxID, bit, state):
        result = []
        command = 'SetBoxCO,' + str(bit) + ',' + str(state) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:设置通用数字输出状态
    *	@param boxID:电箱ID
    *	@param bit : 通用数字输出位
    *	@param state : 设置的通用数字输出状态
    *	@param return: 错误码
    '''

    def HRIF_SetBoxDO(self, boxID, bit, state):
        result = []
        command = 'SetBoxDO,' + str(bit) + ',' + str(state) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:设置模拟量模式
    *	@param boxID:电箱ID
    *	@param bit : 模拟量输出位
    *	@param pattern : 模拟量输出模式,1:电压,2:电流
    *	@param return: 错误码
    '''

    def HRIF_SetBoxAOMode(self, boxID, index, pattern):
        result = []
        command = 'SetBoxAOMode,' + str(index) + ',' + str(pattern) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:设置模拟量输出值和模式
    *	@param boxID:电箱ID
    *	@param bit : 模拟量输出位
    *	@param value : 模拟量输出值
    *	@param pattern : 模拟量输出模式,1:电压,2:电流
    *	@param return: 错误码
    '''

    def HRIF_SetBoxAOVal(self, boxID, index, value, pattern):
        result = []
        command = 'SetBoxAO,' + str(index) + ',' + str(value) + ',' + str(pattern) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:设置末端数字输出状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param bit : 末端数字输出位
    *	@param state : 末端数字输出值
    *	@param return: 错误码
    '''

    def HRIF_SetEndDO(self, boxID, rbtID, bit, state):
        result = []
        command = 'SetEndDO,'
        command += str(rbtID) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 13
    *	@param brief:读取末端数字输入状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param bit : 末端输入位
    *	@param result[0] : 末端输入值
    *	@param return: 错误码
    '''

    def HRIF_ReadEndDI(self, boxID, rbtID, bit, result):
        command = 'ReadEI,'
        command += str(rbtID) + ','
        command += str(bit) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 14
    *	@param brief:读取末端数字输出状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param bit : 末端数字输出位
    *	@param result[0] : 末端数字输出值
    *	@param return: 错误码
    '''

    def HRIF_ReadEndDO(self, boxID, rbtID, bit, result):
        command = 'ReadEO,'
        command += str(rbtID) + ','
        command += str(bit) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 15
    *	@param brief:读取末端模拟量输入状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param bit : 末端模拟量输入位
    *	@param result[0] : 末端模拟量输入值
    *	@param return: 错误码
    '''

    def HRIF_ReadEndAI(self, boxID, rbtID, bit, result):
        command = 'ReadEAI,'
        command += str(rbtID) + ','
        command += str(bit) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 16
    *	@param brief:读取末端按键状态，根据搭载的末端类型，各状态表示含义会有区别
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-3] : 按键1-4状态
    *	@param return: 错误码
    '''

    def HRIF_ReadEndBTN(self, boxID, rbtID, result):
        command = 'ReadEndBTN,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 5 状态读取与设置指令
    #

    '''
    *	@index : 1
    *	@param brief:设置速度百分比
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param vel:速度百分比
    *	@param return: 错误码
    '''

    def HRIF_SetOverride(self, boxID, rbtID, vel):
        result = []
        command = 'SetOverride,'
        command += str(rbtID) + ','
        command += str(vel) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:开启或关闭 Tool 坐标系运动模式
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param state:状态 (0:开启 1:关闭)
    *	@param return: 错误码
    '''

    def HRIF_SetToolMotion(self, boxID, rbtID, state):
        result = []
        command = 'SetTCPMotion,'
        command += str(rbtID) + ','
        command += str(state) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:设置当前负载参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param Mass：质量
    *	@param Center_X：质心X方向偏移
    *	@param Center_Y：质心Y方向偏移
    *	@param Center_Z：质心Z方向偏移
    *	@param return: 错误码
    '''

    def HRIF_SetPayload(self, boxID, rbtID, Mass, Center_X, Center_Y, Center_Z):
        result = []
        command = 'SetPayload,'
        command += str(rbtID) + ','
        command += str(Mass) + ','
        command += str(Center_X) + ','
        command += str(Center_Y) + ','
        command += str(Center_Z) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:设置关节最大运动速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param Joint:轴最大速度
    *	@param return: 错误码
    '''

    def HRIF_SetJointMaxVel(self, boxID, rbtID, Joint):
        result = []
        command = 'SetJointMaxVel,'
        command += str(rbtID) + ','
        for i in range(len(Joint)):
            command += str(Joint[i]) + ','
        command += ';'
        print(command)
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:设置关节最大运动加速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param Joint:速度百分比
    *	@param return: 错误码
    '''

    def HRIF_SetJointMaxAcc(self, boxID, rbtID, Joint):
        result = []
        command = 'SetJointMaxAcc,'
        command += str(rbtID) + ','
        for i in range(len(Joint)):
            command += str(Joint[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:设置直线运动最大速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param MaxVel:最大直线速度
    *	@param return: 错误码
    '''

    def HRIF_SetLinearMaxVel(self, boxID, rbtID, MaxVel):
        result = []
        command = 'SetLinearMaxVel,'
        command += str(rbtID) + ','
        command += str(MaxVel) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:设置直线运动最大加速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param MaxAcc:最大直线速度
    *	@param return: 错误码
    '''

    def HRIF_SetLinearMaxAcc(self, boxID, rbtID, MaxAcc):
        result = []
        command = 'SetLinearMaxAcc,'
        command += str(rbtID) + ','
        command += str(MaxAcc) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:设置最大关节运动范围
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pMax: J1-J6最大范围
    *	@param pMin: J1-J6最小范围
    *	@param return: 错误码
    '''

    def HRIF_SetMaxAcsRange(self, boxID, rbtID, pMax, pMin):
        result = []
        command = 'SetMaxAcsRange,'
        command += str(rbtID) + ','
        command += str(pMax[0])
        command += ','
        command += str(pMax[1])
        command += ','
        command += str(pMax[2])
        command += ','
        command += str(pMax[3])
        command += ','
        command += str(pMax[4])
        command += ','
        command += str(pMax[5])
        command += ','
        command += str(pMin[0])
        command += ','
        command += str(pMin[1])
        command += ','
        command += str(pMin[2])
        command += ','
        command += str(pMin[3])
        command += ','
        command += str(pMin[4])
        command += ','
        command += str(pMin[5])
        command += ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:设置空间最大运动范围
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pMax: X,Y,Z最大范围
    *	@param pMin: X,Y,Z最小范围
    *	@param pUcs: 用户坐标系参数X,Y,Z,RX,RY,RZ
    *	@param return: 错误码
    '''

    def HRIF_SetMaxPcsRange(self, boxID, rbtID, pMax, pMin, pUcs):
        result = []
        command = 'SetMaxPcsRange,'
        command += str(rbtID) + ','
        command += str(pMax[0])
        command += ','
        command += str(pMax[1])
        command += ','
        command += str(pMax[2])
        command += ','
        command += str(180)
        command += ','
        command += str(180)
        command += ','
        command += str(180)
        command += ','
        command += str(pMin[0])
        command += ','
        command += str(pMin[1])
        command += ','
        command += str(pMin[2])
        command += ','
        command += str(-180)
        command += ','
        command += str(-180)
        command += ','
        command += str(-180)
        command += ','
        for i in range(0, 6):
            command += str(pUcs[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:读取关节最大运动速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 轴最大速度
    *	@param return: 错误码
    '''

    def HRIF_ReadJointMaxVel(self, boxID, rbtID, result):
        command = 'ReadJointMaxVel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:读取关节最大运动加速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 轴最大运动加速度
    *	@param return: 错误码
    '''

    def HRIF_ReadJointMaxAcc(self, boxID, rbtID, result):
        command = 'ReadJointMaxAcc,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:读取关节最大运动加加速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 轴最大加加速度
    *	@param return: 错误码
    '''

    def HRIF_ReadJointMaxJerk(self, boxID, rbtID, result):
        command = 'ReadJointMaxJerk,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 13
    *	@param brief:读取直线运动最大速度参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 最大直线速度(单位[mm/ s])
    *	@param result[1] : 最大直线加速度(单位[mm/ s2])
    *	@param result[2] : 最大直线加加速度(单位[mm/ s3])
    *	@param return: 错误码
    '''

    def HRIF_ReadLinearMaxSpeed(self, boxID, rbtID, result):
        command = 'ReadLinearMaxVel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 14
    *	@param brief:读取直线运动最大速度参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 急停错误
    *	@param result[1] : 急停信号
    *	@param result[2] : 安全光幕错误
    *	@param result[3] : 安全光幕信号
    *	@param return: 错误码
    '''

    def HRIF_ReadEmergencyInfo(self, boxID, rbtID, result):
        command = 'ReadEmergencyInfo,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 15
    *	@param brief: 读取当前机器人状态标志
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 运动状态
    *   @param result[1] : 使能状态
    *   @param result[2] : 错误状态
    *   @param result[3] : 错误码
    *   @param result[4] : 错误轴ID
    *   @param result[5] : 抱闸是否打开状态
    *   @param result[6] : 暂停状态
    *   @param result[7] : 急停状态
    *   @param result[8] : 安全光幕状态
    *   @param result[9] : 上电状态
    *   @param result[10] : 连接电箱状态
    *   @param result[11] : WayPoint运动完成状态(BlendingDone)
    *   @param result[12] : 运动命令位置与实际位置是否到位
    *	@param return: 错误码
    '''

    def HRIF_ReadRobotState(self, boxID, rbtID, result):
        command = 'ReadRobotState,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 16
    *	@param brief: 读取当前机器人状态标志
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 运动状态
    *   @param result[1] : 使能状态
    *   @param result[2] : 错误状态
    *   @param result[3] : 错误码
    *   @param result[4] : 错误轴ID
    *   @param result[5] : 抱闸是否打开状态
    *   @param result[6] : 暂停状态
    *   @param result[7] : WayPoint运动完成状态(BlendingDone)
        '''

    def HRIF_ReadRobotFlags(self, boxID, rbtID, result):
        result2 = []
        command = 'ReadRobotState,'
        command += str(rbtID) + ','
        command += ';'
        DataRet = self.g_clients[boxID].sendAndRecv(command, result2)
        for i in range(8):
            result += str(result2[i])
        return DataRet

    '''
    *	@index : 17
    *	@param brief:读取 WayPoint 当前运动 ID 号
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]:当前 ID
    *	@param return: 错误码
    '''

    def HRIF_ReadCurWaypointID(self, boxID, rbtID, result):
        command = 'ReadCurWayPointID,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 18
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param brief:读取错误码
    *	@param result[0-5] : 轴错误码
    *	@param return: 错误码
    '''

    def HRIF_ReadAxiserrorCode(self, boxID, rbtID, result):
        command = 'ReadAxisErrorCode,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 19
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param brief:读取状态机状态
    *	@param result[0] : 当前状态机状态,具体描述见接口说明文档
    *	@param result[1] : 状态机描述
    *	@param return: 错误码
    '''

    def HRIF_ReadCurFSM(self, boxID, rbtID, result):
        command = 'ReadCurFSM,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
      *	@index : 20
      *	@param brief:读取状态机状态
      *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
      *	@param result[0] : 当前状态机状态
      *	@param return: 错误码
    '''

    def HRIF_ReadCurFSMFromCPS(self, boxID, rbtID, result):
        command = 'ReadCurFSM,'
        command += str(rbtID) + ','
        command += ';'
        nRet = self.g_clients[boxID].sendAndRecv(command, result)
        if len(result) < 1:
            return nRet
        strRes = self.dic_FSM.get(int(result[0]))
        result.append(strRes)
        return nRet

    #
    # part 6 位置,速度,电流读取指令
    #

    '''
    *	@index : 1
    *	@param brief:读取当前位置信息
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 迪卡尔坐标
    *	@param result[6-11] : 关节坐标
    *	@param result[12-17] : TCP坐标
    *	@param result[18-23] : 用户坐标
    *	@param return: 错误码
    '''

    def read_pos(self, boxID=0, rbtID=0):
        pose = []
        self.HRIF_ReadActPos(boxID, rbtID, pose)
        pose = [float(num) for num in pose]
        pose = pose[6:12]
        return pose

    def HRIF_ReadActPos(self, boxID, rbtID, result):
        command = 'ReadActPos,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:读取关节命令位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 关节命令位置
    *	@param return: 错误码
    '''

    def HRIF_ReadCmdJointPos(self, boxID, rbtID, result):
        command = 'ReadCmdPos,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:读取关节实际位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 关节实际位置
    *	@param return: 错误码
    '''

    def HRIF_ReadActJointPos(self, boxID, rbtID, result):
        command = 'ReadActACS,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:读取命令 TCP 位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 命令TCP位置
    *	@param return: 错误码
    '''

    def HRIF_ReadCmdTcpPos(self, boxID, rbtID, result):
        command = 'ReadCmdPos,'
        command += str(rbtID) + ','
        command += ';'
        errorCode = self.g_clients[boxID].sendAndRecv(command, result)
        if errorCode != 0:
            return errorCode
        del result[6]
        del result[6]
        del result[6]
        del result[6]
        del result[6]
        del result[6]
        return errorCode

    '''
    *	@index : 5
    *	@param brief: 读取实际 TCP 位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 迪卡尔坐标
    *	@param return: 错误码
    '''

    def HRIF_ReadActTcpPos(self, boxID, rbtID, result):
        retData = []
        errorCode = self.HRIF_ReadActPos(boxID, rbtID, retData)
        if errorCode != 0:
            return errorCode
        for i in range(0, 6):
            result.append(retData[i])
        return errorCode

    '''
    *	@index : 6
    *	@param brief: 读取关节命令速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 关节命令速度
    *	@param return: 错误码
    '''

    def HRIF_ReadCmdJointVel(self, boxID, rbtID, result):
        command = 'ReadCmdJointVel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief: 读取关节实际速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 关节实际速度
    *	@param return: 错误码
    '''

    def HRIF_ReadActJointVel(self, boxID, rbtID, result):
        command = 'ReadActJointVel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief: 读取命令 TCP 速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : TCP命令速度
    *	@param return: 错误码
    '''

    def HRIF_ReadCmdTcpVel(self, boxID, rbtID, result):
        command = 'ReadCmdTcpVel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief: 读取实际 TCP 速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : TCP实际速度
    *	@param return: 错误码
    '''

    def HRIF_ReadActTcpVel(self, boxID, rbtID, result):
        command = 'ReadActTcpVel,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief: 读取关节命令电流
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 关节命令电流 
    *	@param return: 错误码
    '''

    def HRIF_ReadCmdJointCur(self, boxID, rbtID, result):
        command = 'ReadCmdJointCur,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief: 读取关节实际电流
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0-5] : 关节实际电流
    *	@param return: 错误码
    '''

    def HRIF_ReadActJointCur(self, boxID, rbtID, result):
        command = 'ReadActJointCur,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief: 读取 TCP 末端速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 命令速度 
    *	@param result[1] : 实际速度
    *	@param return: 错误码
    '''

    def HRIF_ReadTcpVelocity(self, boxID, rbtID, result):
        command = 'ReadTcpVelocity,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 7 坐标转换计算指令
    #

    '''
    *	@index : 1
    *	@param brief: 四元素转欧拉角
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param dQuaW : W
    *	@param dQuaX : Xi
    *	@param dQuaY : Yj
    *	@param dQuaZ : Zk
    *	@param result[0] : 欧拉角 Rx
    *	@param result[1] : 欧拉角 Ry
    *	@param result[2] : 欧拉角 Rz
    *	@param return: 错误码
    '''

    def HRIF_Quaternion2RPY(self, boxID, rbtID, dQuaW, dQuaX, dQuaY, dQuaZ, result):
        command = 'Quaternion2RPY,'
        command += str(rbtID) + ','
        command += str(dQuaW) + ','
        command += str(dQuaX) + ','
        command += str(dQuaY) + ','
        command += str(dQuaZ) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief: 欧拉角转四元素
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param Rx : 欧拉角 Rx
    *	@param Ry : 欧拉角 Ry
    *	@param Rz : 欧拉角 Rz
    *	@param result[0] : dQuaW
    *	@param result[1] : dQuaX
    *	@param result[2] : dQuaY
    *	@param result[3] : dQuaZ
    *	@param return: 错误码
    '''

    def HRIF_RPY2Quaternion(self, boxID, rbtID, Rx, Ry, Rz, result):
        command = 'RPY2Quaternion,'
        command += str(rbtID) + ','
        command += str(Rx) + ','
        command += str(Ry) + ','
        command += str(Rz) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:逆解,由指定用户坐标系位置和工具坐标系下的迪卡尔坐标计算对应的关节坐标位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param rawPCS : 需要计算逆解的目标迪卡尔位置
    *	@param rawACS : 参考关节坐标,逆解出现多个解时需要根据参考关节坐标选取最终解
    *	@param tcp : 工具坐标
    *	@param ucs : 用户坐标
    *	@return result[0-5] : 用户坐标
    *	@param return: 错误码
    '''

    def HRIF_GetInverseKin(self, boxID, rbtID, rawPCS, rawACS, tcp, ucs, result):
        command = 'PCS2ACS,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(rawPCS[i]) + ','
        for i in range(0, 6):
            command += str(rawACS[i]) + ','
        for i in range(0, 6):
            command += str(tcp[i]) + ','
        for i in range(0, 6):
            command += str(ucs[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:正解，由关节坐标位置计算指定用户坐标系和工具坐标系下的迪卡尔坐标位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param rawACS : 需要计算正解的关节坐标
    *	@param tcp : 工具坐标
    *	@param ucs : 用户坐标
    *	@return result[0-5] : 目标迪卡尔坐标
    *	@param return: 错误码
    '''

    def HRIF_GetForwardKin(self, boxID, rbtID, rawACS, tcp, ucs, result):
        command = 'ACS2PCS,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(rawACS[i]) + ','
        for i in range(0, 6):
            command += str(tcp[i]) + ','
        for i in range(0, 6):
            command += str(ucs[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:由基座坐标系下的坐标位置计算指定用户坐标系和工具坐标系下的迪卡尔坐标位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param Base : 基座坐标系下的迪卡尔坐标位置
    *	@param TCP : 工具坐标
    *	@param UCS : 用户坐标
    *	@return result[0-5] : 目标迪卡尔坐标
    *	@param return: 错误码
    '''

    def HRIF_Base2UcsTcp(self, boxID, rbtID, Base, TCP, UCS, result):
        command = 'Base2UcsTcp,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(Base[i]) + ','
        for i in range(0, 6):
            command += str(TCP[i]) + ','
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:由指定用户坐标系和工具坐标系下的迪卡尔坐标位置计算基座坐标系下的坐标位置
    *	@param boxID:电箱ID
    *	@param UcsTcp : 指定用户坐标系和工具坐标系下的迪卡尔坐标
    *	@param TCP : 工具坐标
    *	@param UCS : 用户坐标
    *	@return result[0-5] : 目标迪卡尔坐标
    *	@param return: 错误码
    '''

    def HRIF_UcsTcp2Base(self, boxID, UcsTcp, TCP, UCS, result):
        command = 'UcsTcp2Base,0,'
        for i in range(0, 6):
            command += str(UcsTcp[i]) + ','
        for i in range(0, 6):
            command += str(TCP[i]) + ','
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:点位加法计算
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pos1 : 空间坐标 1 
    *	@param pos2 : 空间坐标 2 
    *	@return result[0-5] : 计算结果
    *	@param return: 错误码
    '''

    def HRIF_PoseAdd(self, boxID, rbtID, pos1, pos2, result):
        command = 'PoseAdd,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:点位减法计算
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pos1 : 空间坐标 1 
    *	@param pos2 : 空间坐标 2 
    *	@return result[0-5] : 计算结果
    *	@param return: 错误码
    '''

    def HRIF_PoseSub(self, boxID, rbtID, pos1, pos2, result):
        command = 'PoseSub,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:坐标变换,
        组合运算 HRIF_PoseTrans(p1,HRIF_PoseInverse(p2))，得到的就是基坐标系下的 p1,在用户坐标系 p2 下的位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pos1 : 空间坐标 1 
    *	@param pos2 : 空间坐标 2 
    *	@return result[0-5] : 计算结果
    *	@param return: 错误码
    '''

    def HRIF_PoseTrans(self, boxID, rbtID, pos1, pos2, result):
        command = 'PoseTrans,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:坐标逆变换
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pos1 : 空间坐标 1 
    *	@return result[0-5] : 计算结果
    *	@param return: 错误码
    '''

    def HRIF_PoseInverse(self, boxID, rbtID, pos1, result):
        command = 'PoseInverse,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:计算点位距离
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pos1 : 空间坐标 1 
    *	@param pos1 : 空间坐标 2 
    *	@return result[0] : 点位距离 
    *	@return result[1] : 姿态距离
    *	@param return: 错误码
    '''

    def HRIF_PoseDist(self, boxID, rbtID, pos1, pos2, result):
        command = 'CalPointDistance,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:空间位置直线插补计算
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pos1 : 空间坐标 1
    *	@param pos2 : 空间坐标 2
    *	@param alpha : 插补比例 
    *	@return result[0-5] : 计算坐标
    *	@param return: 错误码
    '''

    def HRIF_PoseInterpolate(self, boxID, rbtID, pos1, pos2, alpha, result):
        command = 'PoseInterpolate,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += str(alpha) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 13
    *	@param brief:以轨迹中心旋转计算，p1,p2,p3为旋转前选取的轨迹的特征点，p4,p5,p6为旋转后选取的轨迹的特征点，计算结果表示为旋转特征的用户坐标系
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pose1 : 坐标1(X-Z)
    *	@param pose2 : 坐标2(X-Z)
    *	@param pose3 : 坐标1(X-Z)
    *	@param pose4 : 坐标2(X-Z)
    *	@param pose5 : 坐标1(X-Z)
    *	@param pose6 : 坐标2(X-Z)
    *	@return result[0-5] : 计算坐标
    *	@param return: 错误码
    '''

    def HRIF_PoseDefdFrame(self, boxID, rbtID, pos1, pos2, pos3, pos4, pos5, pos6, result):
        command = 'DefdFrame,'
        command += str(rbtID) + ','
        for i in range(0, 3):
            command += str(pos1[i]) + ','
        for i in range(0, 3):
            command += str(pos2[i]) + ','
        for i in range(0, 3):
            command += str(pos3[i]) + ','
        for i in range(0, 3):
            command += str(pos4[i]) + ','
        for i in range(0, 3):
            command += str(pos5[i]) + ','
        for i in range(0, 3):
            command += str(pos6[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 8 工具坐标与用户坐标读写指令
    #

    '''
    *	@index : 1
    *	@param brief:设置当前工具坐标-不写入配置文件，重启后失效
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param TCP: 工具坐标(x,y,z,Rx,Ry,Rz)
    *	@param return: 错误码
    '''

    def HRIF_SetTCP(self, boxID, rbtID, TCP):
        result = []
        command = 'SetCurTCP,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(TCP[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:设置当前用户坐标-不写入配置文件，重启后失效
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param UCS: 用户坐标(x,y,z,Rx,Ry,Rz)
    *	@param return: 错误码
    '''

    def HRIF_SetUCS(self, boxID, rbtID, UCS):
        result = []
        command = 'SetCurUCS,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:读取当前设置的工具坐标值
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@return result[0-5] : 工具坐标
    *	@param return: 错误码
    '''

    def HRIF_ReadCurTCP(self, boxID, rbtID, result):
        command = 'ReadCurTCP,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:读取当前设置的用户坐标值
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@return result[0-5] : 用户坐标
    *	@param return: 错误码
    '''

    def HRIF_ReadCurUCS(self, boxID, rbtID, result):
        command = 'ReadCurUCS,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:通过名称设置工具坐标列表中的值为当前工具坐标，对应名称为示教器配置页面 TCP 示教的工具名称
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param TcpName: 用户坐标名称
    *	@param return: 错误码
    '''

    def HRIF_SetTCPByName(self, boxID, rbtID, TcpName):
        result = []
        command = 'SetTCPByName,'
        command += str(rbtID) + ','
        command += str(TcpName) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:通过名称设置用户坐标列表中的值为当前用户坐标，对应名称为示教器配置页面用户坐标示教的名称
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param UcsName: 用户坐标名称
    *	@param return: 错误码
    '''

    def HRIF_SetUCSByName(self, boxID, rbtID, UcsName):
        result = []
        command = 'SetUCSByName,'
        command += str(rbtID) + ','
        command += str(UcsName) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:通过名称读取指定 TCP 坐标，对应名称为示教器配置页面 TCP 示教的工具名称
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param TCP: TCP坐标
    *	@return result[0-5] : 工具坐标
    *	@param return: 错误码
    '''

    def HRIF_ReadTCPByName(self, boxID, rbtID, TCP, result):
        command = 'ReadTCPByName,'
        command += str(rbtID) + ','
        command += str(TCP) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:通过名称读取指定 UCS 坐标，对应名称为示教器配置页面用户坐标示教的用户坐标名称
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param UCS: 用户坐标名称
    *	@return result[0-5] : 用户坐标
    *	@param return: 错误码
    '''

    def HRIF_ReadUCSByName(self, boxID, rbtID, UCS, result):
        command = 'ReadUCSByName,'
        command += str(rbtID) + ','
        command += str(UCS) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 9 力控控制指令
    #

    '''
    *	@index : 1
    *	@param brief:设置力传感器状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param state : 1(开启),0(关闭)
    *	@param return: 错误码
    '''

    def HRIF_SetForceControlState(self, boxID, rbtID, state):
        result = []
        command = 'SetForceControlState,'
        command += str(rbtID) + ','
        command += str(state) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:读取探寻状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]:  0 未开启力控
                            1 力控探寻中
                            2 力控探寻完成,保持恒力
                            3 力控拖动开启状态
    *	@param return: 错误码
    '''

    def HRIF_ReadForceControlState(self, boxID, rbtID, result):
        command = 'ReadFTControlState,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:设置力控坐标系方向为 tool 坐标方向模式
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param mode :0 关闭
                     1 开启
    *	@param return: 错误码
    '''

    def HRIF_SetForceToolCoordinateMotion(self, boxID, rbtID, mode, result):
        command = 'SetForceToolCoordinateMotion,'
        command += str(rbtID) + ','
        command += str(mode)
        command += ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:暂停力控运动，仅暂停力控功能，不暂停运动和脚本
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_ForceControlInterrupt(self, boxID, rbtID):
        result = []
        command = 'GrpFCInterrupt,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:继续力控运动，仅继续力控运动功能，不继续运动和脚本
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_ForceControlContinue(self, boxID, rbtID):
        result = []
        command = 'GrpFCContinue,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:力控清零，在原有数据的基础上重新标定力传感器
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_SetForceZero(self, boxID, rbtID):
        result = []
        command = 'SetForceZero,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:设置力控探寻的最大速度
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param MaxLinearVelocity : 直线速度
    *	@param MaxAngularVelocity : 姿态角速度
    *	@param return: 错误码
    '''

    def HRIF_SetMaxSearchVelocities(self, boxID, rbtID, MaxLinearVelocity, MaxAngularVelocity):
        result = []
        command = 'HRSetMaxSearchVelocities,'
        command += str(rbtID) + ','
        command += str(MaxLinearVelocity) + ','
        command += str(MaxAngularVelocity) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:设置力控探寻自由度(0关闭,1开启)
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param X:x方向 
    *	@param Y:y方向
    *	@param Z:z方向
    *	@param Rx:Rx方向
    *	@param Ry:Ry方向
    *	@param Rz:Rz方向
    *	@param return: 错误码
    '''

    def HRIF_SetControlFreedom(self, boxID, rbtID, freedom):
        result = []
        command = 'HRSetControlFreedom,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(freedom[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:设置控制策略
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param strategy: 1:柔顺模式/0:恒力模式
    *	@param return: 错误码
    '''

    def HRIF_SetForceControlStrategy(self, boxID, rbtID, strategy):
        result = []
        command = 'HRSetForceControlStrategy,'
        command += str(rbtID) + ','
        command += str(strategy)
        command += ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:设置力传感器中心相对于法兰盘的安装位置和姿态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param position:X,Y,Z,Rx,Ry,Rz
    *	@param return: 错误码
    '''

    def HRIF_SetFreeDrivePositionAndOrientation(self, boxID, rbtID, position):
        result = []
        command = 'SetFTPosition,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(position[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:设置力控探寻 PID 参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param fP: PID 参数 
    *	@param fI: PID 参数 
    *	@param fD: PID 参数 
    *	@param tP: PID 参数 
    *	@param tI: PID 参数 
    *	@param tD: PID 参数
    *	@param return: 错误码
    '''

    def HRIF_SetPIDControlParams(self, boxID, rbtID, fP, fI, fD, tP, tI, tD):
        result = []
        command = 'HRSetPIDControlParams,'
        command += str(rbtID) + ','
        command += str(fP)
        command += ','
        command += str(fI)
        command += ','
        command += str(fD)
        command += ','
        command += str(tP)
        command += ','
        command += str(tI)
        command += ','
        command += str(tD)
        command += ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:设置惯量控制参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param mass:  惯量控制参数
    *	@param return: 错误码
    '''

    def HRIF_SetMassParams(self, boxID, rbtID, mass):
        result = []
        command = 'HRSetMassParams,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(mass[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 13
    *	@param brief:设置阻尼控制参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param damp:  阻尼控制参数
    *	@param return: 错误码
    '''

    def HRIF_SetDampParams(self, boxID, rbtID, damp):
        result = []
        command = 'HRSetDampParams,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(damp[i]) + ','
        command += ';'
        retData = self.g_clients[boxID].sendAndRecv(command, result)
        return retData

    '''
    *	@index : 14
    *	@param brief:设置刚度(k)控制参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param stiff:  刚度控制参数
    *	@param return: 错误码
    '''

    def HRIF_SetStiffParams(self, boxID, rbtID, stiff):
        result = []
        command = 'HRSetStiffParams,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(stiff[i]) + ','
        command += ';'
        retData = self.g_clients[boxID].sendAndRecv(command, result)
        return retData

    '''
    *	@index : 15
    *	@param brief:设置力控目标力
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param forcegoal:力控目标力
    *	@param return: 错误码
    '''

    def HRIF_SetForceControlGoal(self, boxID, rbtID, forcegoal):
        result = []
        command = 'HRSetControlGoal,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(forcegoal[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 16
    *	@param brief:设置力控目标力和目标距离(力控目标距离暂未启用)
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param forcegoal:力控目标力
    *	@param distance:力控距离
    *	@param return: 错误码
    '''

    def HRIF_SetControlGoal(self, boxID, rbtID, forcegoal, distance):
        result = []
        command = 'HRSetControlGoal,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(forcegoal[i]) + ','
        for i in range(0, 6):
            command += str(distance[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 17
    *	@param brief:设置力控限制范围-力传感器超过此范围后控制器断电
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param max:力最大范围
    *	@param min:力最小范围
    *	@param return: 错误码
    '''

    def HRIF_SetForceDataLimit(self, boxID, rbtID, max, min):
        result = []
        command = 'HRSetForceDataLimit,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(max[i]) + ','
        for i in range(0, 6):
            command += str(min[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 18
    *	@param brief:设置力控形变范围
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param allowDistance:允许最大距离
    *	@param strengthLevel:位置与边界设置偏离距离的幂次项
    *	@param return: 错误码
    '''

    def HRIF_SetForceDistanceLimit(self, boxID, rbtID, allowDistance, strengthLevel):
        result = []
        command = 'HRSetForceDistanceLimit,'
        command += str(rbtID) + ','
        command += str(allowDistance) + ','
        command += str(strengthLevel) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 19
    *	@param brief:设置开启或者关闭力控自由驱动模式
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param state :0(关闭),1(开启)
    *	@param return: 错误码
    '''

    def HRIF_SetForceFreeDriveMode(self, boxID, rbtID, state):
        result = []
        command = ''
        if state == 0:
            command = 'GrpCloseFreeDriver,'
        else:
            command = 'GrpOpenFreeDriver,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 20
    *	@param brief:读取力控标定后数据
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]:x方向力
    *	@param result[1]:y方向力
    *	@param result[2]:z方向力
    *	@param result[3]:Rx方向力
    *	@param result[4]:Ry方向力
    *	@param result[5]:Rz方向力
    *	@param return: 错误码
    '''

    def HRIF_ReadFTCabData(self, boxID, rbtID, result):
        command = 'ReadFTCabData,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 21
    *	@param brief:读取力控原始数据
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]:x方向力
    *	@param result[1]:y方向力
    *	@param result[2]:z方向力
    *	@param result[3]:Rx方向力
    *	@param result[4]:Ry方向力
    *	@param result[5]:Rz方向力
    *	@param return: 错误码
    '''

    def HRIF_ReadFTData(self, boxID, rbtID, result):
        command = 'ReadForceData,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 22
    *	@param brief:越障模式下,设置切向力判断的上下边界
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param Max:上边界,切向力大于该值,激活越障,单位N
    *	@param Min:下边界,切向力小于该值,回归正常力控,单位N
    *	@param Vel:越障抬起线速度，单位mm/s
    *	@param return: 错误码
    '''

    def HRIF_SetTangentForceBounds(self, boxID, rbtID, Max, Min, Vel):
        result = []
        command = 'SetTangentForceBounds,'
        command += str(rbtID) + ','
        command += str(Max) + ','
        command += str(Min) + ','
        command += str(Vel) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 
    *	@param brief:开启关闭力传感器_脚本带配置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param state : 设置力传感器状态
    *	@param FTMode : 控制模式
                     0 : 恒力模式
                     1 : 柔顺模式
    *	@param UCS : Tool
    *	@param vel : vel[0]线速度
                     vel[1]角度速度
    *	@param forces : 目标探寻力x、y、z、Rx、Ry、Rz
    *	@param freedom : 力控探寻自由度X,Y,Z,Rx,Ry,Rz
    *	@param PID : fP,fI,fD,tP,tI,tD
    *	@param Mass : 惯量控制参数
    *	@param Damp : 阻尼控制参数
    *	@param Stiff : 刚度参数x、y、z、Rx、Ry、Rz
    *	@param return: 是否开启力控成功(0成功，1失败)
    '''

    def HRIF_SetScriptForceControlState(self, boxID, rbtID, state, FTMode, UCS, vel, forces, freedom, PID, Mass, Damp,
                                        Stiff):
        result = []
        command = 'SetScriptForceControlState,'
        command += str(rbtID) + ','
        command += str(state) + ','
        command += str(FTMode) + ','
        command += str(UCS) + ','
        for i in range(0, 2):
            command += str(vel[i]) + ','
        for i in range(0, 6):
            command += str(forces[i]) + ','
        for i in range(0, 6):
            command += str(freedom[i]) + ','
        for i in range(0, 6):
            command += str(PID[i]) + ','
        for i in range(0, 6):
            command += str(Mass[i]) + ','
        for i in range(0, 6):
            command += str(Damp[i]) + ','
        for i in range(0, 6):
            command += str(Stiff[i]) + ','
        command += ';'
        retData = self.g_clients[boxID].sendAndRecv(command, result)
        if retData == 0:
            while True:
                command = 'ReadFTControlState,'
                command += str(rbtID) + ','
                command += ';'
                retData = self.g_clients[boxID].sendAndRecv(command, result)
                if state == 1:
                    if int(result[0]) == 2:
                        break
                elif state == 0:
                    if int(result[0]) != 2:
                        time.sleep(0.2)
                        break
                time.sleep(0.2)
        return True

    #
    # part 10 通用运动类控制指令
    #

    '''
    *	@index : 1
    *	@param brief:关节短点动 运动距离 2°，最大速度<10°/s
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param axisId: 关节轴 ID
    *	@param derection: 运动方向(0:负,1:正)
    *	@param return: 错误码
    '''

    def HRIF_ShortJogJ(self, boxID, rbtID, axisId, derection):
        result = []
        command = 'ShortJogJ,'
        command += str(rbtID) + ','
        command += str(axisId) + ','
        command += str(derection) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:空间坐标短点动 运动距离 2mm，最大速度<10mm/s
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pcsId: 坐标系轴 ID
    *	@param derection: 运动方向(0:负,1:正)
    *	@param return: 错误码
    '''

    def HRIF_ShortJogL(self, boxID, rbtID, pcsId, derection):
        result = []
        command = 'ShortJogL,'
        command += str(rbtID) + ','
        command += str(pcsId) + ','
        command += str(derection) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:关节长点动，最大运动速度<10°/s
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param axisId: 关节轴 ID
    *	@param derection: 运动方向(0:负,1:正)
    *	@param state: 0:关闭,1:开启
    *	@param return: 错误码
    '''

    def HRIF_LongJogJ(self, boxID, rbtID, axisId, derection, state):
        result = []
        command = 'LongJogJ,'
        command += str(rbtID) + ','
        command += str(axisId) + ','
        command += str(derection) + ','
        command += str(state) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:空间长点动,最大运动速度<50mm/s
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pcsId: 坐标系轴 ID
    *	@param derection: 运动方向(0:负,1:正)
    *	@param state: 0:关闭,1:开启
    *	@param return: 错误码
    '''

    def HRIF_LongJogL(self, boxID, rbtID, pcsId, derection, state):
        result = []
        command = 'LongJogL,'
        command += str(rbtID) + ','
        command += str(pcsId) + ','
        command += str(derection) + ','
        command += str(state) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    ''' 
    *	@index : 5
    *	@param brief:长点动继续指令，当开始长点动之后，要按 500 毫秒或更短时间为时间周期发送一次该指令，否则长点动会停止
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_LongMoveEvent(self, boxID, rbtID):
        result = []
        command = 'LongMoveEvent,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    ''' 
    *	@index : 6
    *	@param brief:判断机器人是否处于运动状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]: 1:运动完成,0：运动未完成
    *	@param return: 错误码
    '''

    def HRIF_IsMotionDone(self, boxID, rbtID, result):
        ret = []
        errorCode = self.HRIF_ReadRobotState(boxID, rbtID, ret)
        if errorCode != 0:
            return errorCode
        result.append(ret[11] == "1" and ret[0] == "0")
        return errorCode

    ''' 
    *	@index : 7
    *	@param brief:判断路点是否运动完成
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0]: 1:运动完成,0：运动未完成
    *	@param return: 错误码
    '''

    def HRIF_IsBlendingDone(self, boxID, rbtID, result):
        ret = []
        errorCode = self.HRIF_ReadRobotState(boxID, rbtID, ret)
        if errorCode != 0:
            return errorCode
        result.append(ret[11])
        return errorCode

    '''
    *	@index : 8
    *	@param brief:执行路点运动(HRIF_WayPointEx 与 HRIF_WayPoint 区别在于 HRIF_WayPointEx 需要设置工具坐标与用户坐标具体的值，而HRIF_WayPoint 使用示教器示教的对应工具坐标与用户坐标名称)
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param type : 运动类型(0:关节运动,1:空间运动)
    *	@param points : 空间目标位置
    *	@param RawACSpoints : 目标关节位置
    *	@param tcp : 工具坐标值
    *	@param ucs : 用户坐标值
    *	@param speed : 运动速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param acc : 运动加速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param radius : 是过渡半径,单位毫米
    *	@param isJoint : 是否使用关节角度作为目标点,如果type==0,则isJointt有起作用
    *	@param isSeek,bit,state:探寻参数,当ieek为1,则开启探寻,这时电箱的DO bit位为state时,就停止运动,否则运动到目标点再停止
    *	@param cmdID:当前路点ID,可以自定义,也可以按顺序设置为“1”,“2”,“3”
    *	@param return: 错误码
    '''

    def HRIF_WayPointEx(self, boxID, rbtID, type, points, RawACSpoints, tcp, ucs, speed, acc, radius, isJoint, isSeek,
                        bit, state, cmdID):
        result = []
        command = 'WayPointEx,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(points[i]) + ','
        for i in range(0, 6):
            command += str(RawACSpoints[i]) + ','
        for i in range(0, 6):
            command += str(ucs[i]) + ','
        for i in range(0, 6):
            command += str(tcp[i]) + ','
        command += str(speed) + ','
        command += str(acc) + ','
        command += str(radius) + ','
        command += str(type) + ','
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += cmdID + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:路点运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param type : 运动类型(0:关节运动,1:直线运动)
    *	@param points : 目标迪卡尔位置
    *	@param RawACSpoints : 目标关节位置
    *	@param tcp : 工具坐标名称
    *	@param ucs : 用户坐标名称
    *	@param speed : 运动速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param Acc : 运动加速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param radius : 是过渡半径,单位毫米
    *	@param isJoint : 是否使用关节角度作为目标点,如果type==0,则isJoint有起作用
    *	@param isSeek,bit,state:探寻参数,当isSeek为1,则开启探寻,这时电箱的DO bit位为state时,就停止运动,否则运动到目标点再停止
    *	@param cmdID:当前路点ID,可以自定义,也可以按顺序设置为“1”,“2”,“3”
    *	@param return: 错误码
    '''

    def HRIF_WayPoint(self, boxID, rbtID, type, points, RawACSpoints, tcp, ucs, speed, Acc, radius, isJoint, isSeek,
                      bit, state, cmdID):
        result = []
        command = 'WayPoint,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(points[i]) + ','
        for i in range(0, 6):
            command += str(RawACSpoints[i]) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += str(type) + ','
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:路点运动(HRIF_WayPoint2 新增直线与圆弧过渡不减速功能，HRIF_WayPoint 只有直线与直线之间有过渡)
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param type : 运动类型(0:关节运动,1:直线运动,2:圆弧运动)
    *	@param EndPos : 空间目标位置(type=0 并 isJoint=1:无效)
                        (type=0 并 isJoint=0:用此空间坐标作为目标位置,通过逆解计算得到关节坐标为目标关节坐标)
    *	@param AuxPos : 空间目标位置(type=0或1时无效,type=2时做为圆弧的经过位置)
    *	@param AcsPos : 空间目标位置(type=0 并 isJoint=1:使用此关节坐标作为目标关节坐标)
                        (type=0 并 isJoint=0:此关节坐标仅作为计算逆解时选解的参考关节坐标)
                        (type=1 或 2 :无效)

    *	@param Tcp : 工具坐标名称
    *	@param Ucs : 用户坐标名称
    *	@param Vel : 运动速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param Acc : 运动加速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param Radius : 是过渡半径,单位毫米
    *	@param isJoint : 是否使用关节角度作为目标点,如果type==0,则isJoint有起作用
    *	@param isSeek,bit,state:探寻参数,当isSeek为1,则开启探寻,这时电箱的DO bit位为state时,就停止运动,否则运动到目标点再停止
    *	@param cmdID:当前路点ID,可以自定义,也可以按顺序设置为“1”,“2”,“3”
    *	@param return: 错误码
    '''

    def HRIF_WayPoint2(self, boxID, rbtID, EndPos, AuxPos, cmdID, Tcp, Ucs, Vel, Acc, Radius, type, isJoint, isSeek,
                       bit, state, AcsPos):
        result = []
        command = 'WayPoint2,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(EndPos[i]) + ','
        for i in range(0, 6):
            command += str(AuxPos[i]) + ','
        command += str(cmdID) + ','
        command += str(Tcp) + ','
        command += str(Ucs) + ','
        command += str(Vel) + ','
        command += str(Acc) + ','
        command += str(Radius) + ','
        command += str(type) + ','
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        for i in range(0, 6):
            command += str(AcsPos[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:机器人运动到指定的角度坐标位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param points : 目标迪卡尔位置
    *	@param RawACSpoints : 目标关节位置
    *	@param tcp : 工具坐标名称
    *	@param ucs : 用户坐标名称
    *	@param speed : 运动速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param Acc : 运动加速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param radius : 是过渡半径,单位毫米
    *	@param isJoint : 是否使用关节角度作为目标点,如果type==0,则isJoint有起作用
    *	@param isSeek,bit,state:探寻参数,当isSeek为1,则开启探寻,这时电箱的DO bit位为state时,就停止运动,否则运动到目标点再停止
    *	@param cmdID:当前路点ID,可以自定义,也可以按顺序设置为“1”,“2”,“3”
    *	@param return: 错误码
    '''

    def moveJ_robot(self, target_joint, boxID=0, rbtID=0, speed=30, acceleration=50, radius=0, isJoint=1):
        """
        通过关节插补方式移动机器人到目标位姿。

        参数:
            target_joint (list): 目标位姿，包含6个关节值。
            boxID (int): 盒子ID，默认值为0。
            rbtID (int): 机器人ID，默认值为0。
            speed (int): 移动速度，默认值为50。
            acceleration (int): 加速度，默认值为500。
            radius (int): 路径半径，默认值为0。
            isJoint (int): 是否为关节插补，默认值为1（关节插补）。

        返回:
            None
        """
        ucs = "Base"  # 坐标系
        ret = self.HRIF_MoveJ(
            boxID=boxID,
            rbtID=rbtID,
            points=[0, 0, 0, 0, 0, 0],
            RawACSpoints=target_joint,
            tcp='TCP',
            ucs=ucs,
            speed=speed,
            Acc=acceleration,
            radius=radius,
            isJoint=isJoint,
            isSeek=0,
            bit=0,
            state=0,
            cmdID=1
        )

        if ret == 0:
            # 等待运动完成
            while True:
                motion_done_result = []
                motion_done = self.HRIF_IsMotionDone(boxID, rbtID, motion_done_result)
                if motion_done == 0 and motion_done_result and motion_done_result[0] == 1:
                    break  # 运动完成
                elif motion_done < 0:
                    print(f"运动过程中出错，错误码: {motion_done}")
                    break
                time.sleep(0.5)  # 等待一段时间再次检查
        else:
            print(f"机器人运动失败，错误码: {ret}")

    def HRIF_MoveJ(self, boxID, rbtID, points, RawACSpoints, tcp, ucs, speed, Acc, radius, isJoint, isSeek, bit, state,
                   cmdID):
        result = []
        command = 'WayPoint,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(points[i]) + ','
        for i in range(0, 6):
            command += str(RawACSpoints[i]) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += '0,'
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:机器人直线运动到指定的空间坐标位置
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param points : 目标迪卡尔位置
    *	@param RawACSpoints : 目标关节位置
    *	@param tcp : 工具坐标名称
    *	@param ucs : 用户坐标名称
    *	@param speed : 运动速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param Acc : 运动加速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param radius : 是过渡半径,单位毫米
    *	@param isSeek,bit,state:探寻参数,当isSeek为1,则开启探寻,这时电箱的DO bit位为state时,就停止运动,否则运动到目标点再停止
    *	@param cmdID:当前路点ID,可以自定义,也可以按顺序设置为“1”,“2”,“3”
    *	@param return: 错误码
    '''

    def move_robot(self, target_pose, boxID=0, rbtID=0, speed=100, acceleration=500, radius=0):
        ucs = "Base"  # 坐标系
        ret = self.HRIF_MoveL(boxID, rbtID, points=target_pose, RawACSpoints=target_pose, tcp="TCP", ucs=ucs,
                              speed=speed, Acc=acceleration, radius=radius, isSeek=0, bit=0, state=1, cmdID=1)
        if ret == 0:
            # 等待运动完成
            while True:
                motion_done_result = []
                motion_done = self.HRIF_IsMotionDone(boxID, rbtID, motion_done_result)
                if motion_done == 0 and motion_done_result and motion_done_result[0] == 1:
                    break  # 运动完成
                elif motion_done < 0:
                    print(f"运动过程中出错，错误码: {motion_done}")
                    break
                time.sleep(0.5)  # 等待一段时间再次检查
        else:
            print(f"机器人运动失败，错误码: {ret}")
        return ret

    def HRIF_MoveL(self, boxID, rbtID, points, RawACSpoints, tcp, ucs, speed, Acc, radius, isSeek, bit, state, cmdID):
        result = []
        command = 'WayPoint,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(points[i]) + ','
        for i in range(0, 6):
            command += str(RawACSpoints[i]) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += '1,'
        command += '0,'
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 13
    *	@param brief:圆弧轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param StartPoint : 圆弧开始位置
    *	@param AuxPoint : 圆弧经过位置
    *	@param EndPoint : 圆弧结束位置
    *	@param fixedPosure : 0:不使用固定姿态,1:使用固定姿态
    *	@param nMoveCType : 0:圆弧运动,1:整圆运动
    *	@param nRadLen : 当nMoveCType=0时该参数无效,由三个点确定圆弧轨迹
                         当nMoveCType=1时,该参数为整圆的圈数
    *	@param speed : 运动速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param Acc : 运动加速度,速度单位是毫米每秒,度每秒,加速度毫米每秒平方,度每秒平方
    *	@param radius : 是过渡半径,单位毫米
    *	@param tcp : 目标所在的工具坐标名称
    *	@param ucs : 目标所在的用户坐标名称
    *	@param cmdID:当前路点ID,可以自定义,也可以按顺序设置为“1”,“2”,“3”
    *	@param return: 错误码
    '''

    def HRIF_MoveC(self, boxID, rbtID, StartPoint, AuxPoint, EndPoint, fixedPosure, nMoveCType, nRadLen, speed, Acc,
                   radius, tcp, ucs, cmdID):
        result = []
        command = 'MoveC,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(StartPoint[i]) + ','
        for i in range(0, 6):
            command += str(AuxPoint[i]) + ','
        for i in range(0, 6):
            command += str(EndPoint[i]) + ','
        command += str(fixedPosure) + ','
        command += str(nMoveCType) + ','
        command += str(nRadLen) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 14
    *	@brief: Z型轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param StartPoint : 开始位置
    *	@param EndPoint : 结束位置
    *	@param PlanePoint : 确定平面点位置
    *	@param Speed : 速度
    *	@param Acc : 加速度
    *	@param WIdth : 宽度
    *	@param Density : 密度
    *	@param EnableDensity : 是否使用密度(0:不使用,1:使用)
    *	@param EnablePlane : 是否使用平面点(0:不使用,1:使用)
    *	@param EnableWaiTime : 是否开启转折点等待时间(0:不使用,1:使用)
    *	@param PosiTime : 正向转折点等待时间ms
    *	@param NegaTime : 负向转折点等待时间ms
    *	@param Radius : 过渡半径
    *	@param tcp : 工具坐标名称
    *	@param ucs : 用户坐标名称
    *	@param cmdID : 命令ID
    *	@param return: 错误码
    '''

    def HRIF_MoveZ(self, boxID, rbtID, StartPoint, EndPoint, PlanePoint, Speed, Acc, WIdth, Density, EnableDensity,
                   EnablePlane, EnableWaiTime, PosiTime, NegaTime, Radius, tcp, ucs, cmdID):
        result = []
        command = 'MoveZ,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(StartPoint[i]) + ','
        for i in range(0, 6):
            command += str(EndPoint[i]) + ','
        for i in range(0, 6):
            command += str(PlanePoint[i]) + ','
        command += str(Speed) + ','
        command += str(Acc) + ','
        command += str(WIdth) + ','
        command += str(Density) + ','
        command += str(EnableDensity) + ','
        command += str(EnablePlane) + ','
        command += str(EnableWaiTime) + ','
        command += str(PosiTime) + ','
        command += str(NegaTime) + ','
        command += str(Radius) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 15
    *	@brief: 椭圆型轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param dP1 : 示教位置1
    *	@param dP2 : 示教位置2
    *	@param dP3 : 示教位置3
    *	@param dP4 : 示教位置4
    *	@param dP5 : 示教位置5
    *	@param nOrientMode : 运动模式，0：椭圆圆弧 1：整个椭圆
    *	@param nMoveType : 运动类型，0:不使用固定姿态 1:使用固定姿态
    *	@param dArcLength : 弧长,单位°
    *	@param dVelocity : 运动最大速度，X，Y，Z单位[mm/s]，Rx，Ry，Rz单位[°/s]
    *	@param dAcc : 运动最大加速度，X，Y，Z单位[mm/s2]，Rx，Ry，Rz单位[°/s2]
    *	@param Radius : 过渡半径，单位[mm]
    *	@param tcp : 目标空间坐标所处的工具坐标系名称，与示教器页面的名称对应，当nIsUseJoint=1时无效，可使用默认名称"TCP"
    *	@param ucs : 目标空间坐标所处的用户坐标系名称，与示教器页面的名称对应，当nIsUseJoint=1时无效，可使用默认名称"Base"
    *	@param cmdID : 命令ID
    *	@param return: 错误码
    '''

    def HRIF_MoveE(self, boxID, rbtID, dP1, dP2, dP3, dP4, dP5, nOrientMode, nMoveType, dArcLength, dVelocity, dAcc,
                   Radius, tcp, ucs, cmdID):
        result = []
        command = 'MoveE,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(dP1[i]) + ','
        for i in range(0, 6):
            command += str(dP2[i]) + ','
        for i in range(0, 6):
            command += str(dP3[i]) + ','
        for i in range(0, 6):
            command += str(dP4[i]) + ','
        for i in range(0, 6):
            command += str(dP5[i]) + ','
        command += str(nOrientMode) + ','
        command += str(nMoveType) + ','
        command += str(dArcLength) + ','
        command += str(dVelocity) + ','
        command += str(dAcc) + ','
        command += str(Radius) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 16
    *	@brief: 螺旋轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param dSpiralIncrement : 螺旋运动每圈增量半径，单位[mm]
    *	@param dSpiralDiameter : 螺旋运动结束半径，单位[mm]
    *	@param dVelocity : 运动最大速度，单位[°/s]
    *	@param dAcc : 运动最大加速度，单位[°/s2]
    *	@param dRadius : 过渡半径，单位[mm]
    *	@param sTcpName : 目标空间坐标所处的工具坐标系名称，与示教器页面的名称对应，当nIsUseJoint=1时无效，可使用默认名称"TCP"
    *	@param sUcsName : 目标空间坐标所处的用户坐标系名称，与示教器页面的名称对应，当nIsUseJoint=1时无效，可使用默认名称"Base"
    *	@param cmdID : 命令ID
    *	@param return: 错误码
    '''

    def HRIF_MoveS(self, boxID, rbtID, dSpiralIncrement, dSpiralDiameter, dVelocity, dAcc, dRadius, sTcpName, sUcsName,
                   cmdID):
        result = []
        command = 'MoveS,'
        command += str(rbtID) + ','
        command += str(dSpiralIncrement) + ','
        command += str(dSpiralDiameter) + ','
        command += str(dVelocity) + ','
        command += str(dAcc) + ','
        command += str(dRadius) + ','
        command += str(sTcpName) + ','
        command += str(sUcsName) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 17
    *	@brief: 关节相对运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param nAxis : 轴ID
    *	@param nDirection : 方向
    *	@param dDistance : 运动距离
    '''

    def HRIF_MoveRelJ(self, boxID, rbtID, nAxis, nDirection, dDistance):
        result = []
        command = 'MoveRelJ,'
        command += str(rbtID) + ','
        command += str(nAxis) + ','
        command += str(nDirection) + ','
        command += str(dDistance) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 18
    *	@brief: 空间相对运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param nAxis : 轴ID
    *	@param nDirection : 方向
    *	@param dDistance : 运动距离
    *   @param nToolMotion：运动坐标类型
    '''

    def HRIF_MoveRelL(self, boxID, rbtID, nAxis, nDirection, dDistance, nToolMotion):
        result = []
        command = 'MoveRelL,'
        command += str(rbtID) + ','
        command += str(nAxis) + ','
        command += str(nDirection) + ','
        command += str(dDistance) + ','
        command += str(nToolMotion) + ','
        command += ';'
        print(command)
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 19
    *	@brief: 路点相对运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param nType : 运动类型
    *	@param nPointList : 是否使用点位列表点位
    *	@param Pos : 空间位置
    *   @param rawACT : 关节位置
    *   @param nrelMoveType : 相对运动类型
    *   @param nAxisMask : 各轴是否运动
    *   @param dTarget : 运动距离
    *   @param sTcpName : 工具坐标系名称
    *   @param sUcsName : 用户坐标系名称
    *   @param dVelocity : 速度
    *   @param dAcc : 加速度
    *   @param dRadius : 过渡半径
    *   @param nIsUseJoint : 是否使用关节坐标
    *   @param nIsSeek :是否检测 DI 停止
    *   @param nIOBit : 检测的 DI 索引
    *   @param nIOState : 检测的 DI 状态
    *   @param strCmdID : 命令ID
    '''

    def HRIF_WayPointRel(self, boxID, rbtID, nType, nPointList, Pos, rawACT, nrelMoveType, nAxisMask, dTarget, sTcpName,
                         sUcsName, dVelocity, dAcc, dRadius, nIsUseJoint, nIsSeek, nIOBit, nIOState, strcmdID):
        result = []
        command = 'WayPointRel,'
        command += str(rbtID) + ','
        command += str(nType) + ','
        command += str(nPointList) + ','
        for i in range(0, 6):
            command += str(Pos[i]) + ','
        for i in range(0, 6):
            command += str(rawACT[i]) + ','
        command += str(nrelMoveType) + ','
        for i in range(0, 6):
            command += str(nAxisMask[i]) + ','
        for i in range(0, 6):
            command += str(dTarget[i]) + ','
        command += str(sTcpName) + ','
        command += str(sUcsName) + ','
        command += str(dVelocity) + ','
        command += str(dAcc) + ','
        command += str(dRadius) + ','
        command += str(nIsUseJoint) + ','
        command += str(nIsSeek) + ','
        command += str(nIOBit) + ','
        command += str(nIOState) + ','
        command += str(strcmdID) + ',;'
        print(command)
        return self.g_clients[boxID].sendAndRecv(command, result)

    # part 11 连续轨迹运动类控制指令
    #

    '''
    *	@index : 1
    *	@param brief:初始化关节连续轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param speedRatio : 运动速度
    *	@param radius : 过渡半径
    *	@param return: 错误码
    '''

    def HRIF_StartPushMovePathJ(self, boxID, rbtID, trackName, speedRatio, radius):
        result = []
        command = 'StartPushMovePath,'
        command += str(rbtID) + ','
        command += trackName + ','
        command += str(speedRatio) + ','
        command += str(radius) + ','
        command += ';'
        print(command)
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:下发轨迹点位
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param paramsJ : 关节点位(6)
    *	@param return: 错误码
    '''

    def HRIF_PushMovePathJ(self, boxID, rbtID, trackName, paramsJ):
        result = []
        command = 'PushMovePathJ,'
        command += str(rbtID) + ','
        command += trackName
        command += ','
        for i in range(0, 6):
            command += str(paramsJ[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:轨迹下发完成,开始计算轨迹
			调用pushMovePathJ,一般情况下点位数量需要>4
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param return: 错误码
    '''

    def HRIF_EndPushMovePathJ(self, boxID, rbtID, trackName):
        result = []
        command = 'EndPushMovePath,'
        command += str(rbtID) + ','
        command += trackName + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:执行轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param return: 错误码
    '''

    def HRIF_MovePathJ(self, boxID, rbtID, trajectName):
        result = []
        command = 'MovePath,'
        command += str(rbtID) + ','
        command += str(trajectName) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:读取当前的轨迹状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param result[0] :  
                    0 : 轨迹未示教
                    1 : 轨迹示教中
                    2 : 轨迹计算中
                    3 : 轨迹完成计算
                    4 : 轨迹完成示教
                    5 : 轨迹计算错误
    *	@param return: 错误码
    '''

    def HRIF_ReadMovePathJState(self, boxID, rbtID, trackName, result):
        command = 'ReadMovePathState,'
        command += str(rbtID) + ','
        command += trackName + ','
        command += ';'
        retData = self.g_clients[boxID].sendAndRecv(command, result)
        return retData

    '''
    *	@index : 6
    *	@param brief:修改轨迹名称
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param newName : 新轨迹名称
    *	@param return: 错误码
    '''

    def HRIF_UpdateMovePathJName(self, boxID, rbtID, trackName, newName):
        result = []
        command = 'UpdateMovePathName,'
        command += str(rbtID) + ','
        command += trackName + ','
        command += newName + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:删除指定轨迹
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param return: 错误码
    '''

    def HRIF_DelMovePathJ(self, boxID, rbtID, trackName):
        result = []
        command = 'DelMovePath,'
        command += str(rbtID) + ','
        command += trackName + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 8
    *	@param brief:读取当前的轨迹运动进度(此接口仅对MovePathL有效，对MovePathJ无效)
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *   @param result[0] : 0:未完成轨迹
                            1:已完成轨迹
    *   @param result[1] : 当前运动到的点位数量
    *	@param return: 错误码
    '''

    def HRIF_ReadTrackProcess(self, boxID, rbtID, result):
        command = 'ReadSoftMotionProgress,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 9
    *	@param brief:初始化空间轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param vel : 运动速度
    *	@param acc : 运动加速度
    *	@param jerk : 运动加加速度
    *	@param ucs : 指定轨迹所在的用户坐标系名称
    *	@param tcp : 指定轨迹所在的工具坐标值名称
    *	@param return: 错误码
    '''

    def HRIF_InitMovePathL(self, boxID, rbtID, trackName, vel, acc, jerk, ucs, tcp):
        result = []
        command = 'InitMovePathL,'
        command += str(rbtID) + ','
        command += trackName + ','
        command += str(vel) + ','
        command += str(acc) + ','
        command += str(jerk) + ','
        command += ucs + ','
        command += tcp + ','
        command += ';'
        print(command)
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 10
    *	@param brief:下发轨迹点位
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param paramPcs : 空间点位
    *	@param return: 错误码
    '''

    def HRIF_PushMovePathL(self, boxID, rbtID, trackName, paramPcs):
        result = []
        command = 'PushMovePathL,'
        command += str(rbtID) + ','
        command += trackName + ','
        for i in range(0, 6):
            command += str(paramPcs[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 11
    *	@param brief:批量下发轨迹点位，调用一次可下发多个点位数据
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param moveType : 运动类型-MovePathJ可以共用 0(MovePathJ)/1(MovePathL)
    *	@param pointsSize : 轨迹点位数量
    *	@param points : 轨迹点位
    *	@param return: 错误码
    '''

    def HRIF_PushMovePaths(self, boxID, rbtID, trackName, moveType, pointsSize, points):
        result = []
        command = 'PushMovePaths,'
        command += str(rbtID) + ','
        command += trackName
        command += ','
        command += str(moveType)
        command += ','
        command += str(pointsSize)
        command += ','
        for pos in points:
            command += str(pos) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 12
    *	@param brief:执行空间坐标轨迹运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param trackName : 轨迹名称
    *	@param return: 错误码
    '''

    def HRIF_MovePathL(self, boxID, rbtID, trackName):
        result = []
        command = 'MovePathL,'
        command += str(rbtID) + ','
        command += trackName
        command += ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 13
    *	@param brief:设置MovePath速度比
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param MovePathOverride : 设置速度比
    *	@param return: 错误码
    '''

    def HRIF_SetMovePathOverride(self, boxID, rbtID, MovePathOverride):
        result = []
        command = 'SetMovePathOverride,'
        command += str(rbtID) + ','
        command += str(MovePathOverride)
        command += ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 12 Servo 运动类控制指令
    #

    '''
    *	@index : 1
    *	@param brief:启动机器人在线控制(servoJ 或 servoP)时,设定位置固定更新的周期和前瞻时间
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param servoTime : 固定更新的周期 s
    *	@param lookaheadTime : 前瞻时间 s
    *	@param return: 错误码
    '''

    def HRIF_StartServo(self, boxID, rbtID, servoTime, lookaheadTime):
        result = []
        command = 'StartServo,'
        command += str(rbtID) + ','
        command += str(servoTime) + ',' + str(lookaheadTime) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:在线关节位置命令控制，以 StartServo 设定的固定更新时间发送关节位置，机器人将实时的跟踪关节位置指令
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param dACSd : 关节点位
    *	@param return: 错误码
    '''

    def HRIF_PushServoJ(self, boxID, rbtID, dACS):
        result = []
        command = 'PushServoJ,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(dACS[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:在线末端TCP位置命令控制,以 StartServo 设定的固定更新时间发送 TCP 位置，机器人将实时的跟踪目标 TCP 位置逆运算转换后的关节位置指令
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param pose : 更新的目标迪卡尔坐标位置
    *	@param ucs : 目标位置对应的UCS
    *	@param tcp : 目标位置对应的TCP
    *	@param return: 错误码
    '''

    def HRIF_PushServoP(self, boxID, rbtID, pose, ucs, tcp):
        result = []
        command = 'PushServoP,'
        command += str(rbtID) + ','
        for i in range(0, 6):
            command += str(pose[i]) + ','
        for i in range(0, 6):
            command += str(ucs[i]) + ','
        for i in range(0, 6):
            command += str(tcp[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief:初始化在线控制模式，清空缓存点位,ServoEsJ
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param return: 错误码
    '''

    def HRIF_InitServoEsJ(self, boxID, rbtID):
        result = []
        command = 'InitServoEsJ,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 5
    *	@param brief:启动在线控制模式，设定位置固定更新的周期和前瞻时间，开始运动
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param dServoTime : 更新周期
    *	@param dLookaheadTime : 前瞻时间
    *	@param return: 错误码
    '''

    def HRIF_StartServoEsJ(self, boxID, rbtID, dServoTime, dLookaheadTime):
        result = []
        command = 'StartServoEsJ,'
        command += str(rbtID) + ','
        command += str(dServoTime) + ','
        command += str(dLookaheadTime) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 6
    *	@param brief:批量下发在线控制点位,每个点位下发频率由固定更新的周期确定
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param nPointSize : 点位数量
    *	@param sPoints : 点位信息
    *	@param return: 错误码
    '''

    def HRIF_PushServoEsJ(self, boxID, rbtID, nPointSize, sPoints):
        result = []
        command = 'PushServoEsJ,'
        command += str(rbtID) + ','
        command += str(nPointSize) + ','
        for i in range(len(sPoints)):
            command += str(sPoints[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 7
    *	@param brief:读取当前是否可以继续下发点位信息，循环读取间隔>20ms
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param result[0] : 点位状态,0：允许下发点位 1：不允许下发点位
    *	@param return: 错误码
    '''

    def HRIF_ReadServoEsJState(self, boxID, rbtID, result):
        command = 'ReadServoEsJState,'
        command += str(rbtID) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 13 相对跟踪运动类控制指令
    #

    '''
    *	@index : 1
    *	@param brief:启动机器人在线控制(servoJ 或 servoP)时,设定位置固定更新的周期和前瞻时间
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param state : 跟踪状态(0:关闭相对跟踪运动 1:开启相对跟踪运动)
    *	@param distance : 相对跟踪运动保持的相对距离
    *	@param dAwayVelocity: 相对跟踪的运动的远离速度
    *	@param dGobackVelocity: 相对跟踪的运动的返回速度
    *	@param return: 错误码
    '''

    def HRIF_SetMoveTraceParams(self, boxID, rbtID, state, distance, dAwayVelocity, dGobackVelocity):
        result = []
        command = 'SetMoveTraceParams,'
        command += str(rbtID) + ','
        command += str(state) + ','
        command += str(distance) + ','
        command += str(dAwayVelocity) + ','
        command += str(dGobackVelocity) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:设置相对跟踪运动初始化参数
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param dK,dB: 计算公式y = dK * x + dB
    *	@param maxLimit: 激光传感器检测距离最大值
    *	@param minLinit: 激光传感器检测距离最小值
    *	@param return: 错误码
    '''

    def HRIF_SetMoveTraceInitParams(self, boxID, rbtID, dK, dB, maxLimit, minLinit):
        result = []
        command = 'SetMoveTraceInitParams,'
        command += str(rbtID) + ','
        command += str(dK) + ','
        command += str(dB) + ','
        command += str(maxLimit) + ','
        command += str(minLinit) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 3
    *	@param brief:设置相对跟踪运动的跟踪探寻方向
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param direction: 跟踪探寻方向(x,y,z无效,可设置0)(Rx,Ry,Rz单位[°])
    *	@param return: 错误码
    '''

    def HRIF_SetMoveTraceUcs(self, boxID, rbtID, direction):
        result = []
        command = 'SetMoveTraceUcs,'
        command += str(rbtID) + ','
        for i in range(len(direction)):
            command += str(direction[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 4
    *	@param brief: 设置传送带跟踪运动状态
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param state: 0:关闭 1:开启
    *	@param return: 错误码
    '''

    def HRIF_SetTrackingState(self, boxID, rbtID, state):
        result = []
        command = 'SetTrackingState,'
        command += str(rbtID) + ','
        command += str(state) + ',;'
        return self.g_clients[boxID].sendAndRecv(command, result)

    #
    # part 14 其他指令
    #

    '''
    *	@index : 1
    *	@param brief:执行插件 app 命令
    *	@param boxID:电箱ID
    *	@param name :插件名称
    *	@param param:插件指令及参数
    *	@param return: 错误码
    '''

    def HRIF_HRAppCmd(self, boxID, name, cmd, param, result):
        command = 'HRAppCmd,'
        command += str(name) + ','
        command += str(cmd) + ','
        for i in range(len(param)):
            command += str(param[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 2
    *	@param brief:写末端连接的 modbus 从站寄存器
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param nSlaveID : 从站ID
    *	@param nFunction : 功能码
    *	@param nRegAddr : 寄存器地址
    *	@param nRegCount : 寄存器数量
    *	@param data : 寄存器数据
    *	@param return: 错误码
    '''

    def HRIF_WriteEndHoldingRegisters(self, boxID, rbtID, nSlaveID, nFunction, nRegAddr, nRegCount, data):
        result = []
        command = 'WriteHoldingRegisters,'
        command += str(rbtID) + ','
        command += str(nSlaveID) + ','
        command += str(nFunction) + ','
        command += str(nRegAddr) + ','
        command += str(nRegCount) + ','
        if nRegCount != len(data):
            return ['-1']
        for i in range(nRegCount):
            command += str(data[i]) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)

    '''
    *	@index : 
    *	@param brief:读取末端Modbus寄存器
    *	@param boxID:电箱ID
    *	@param rbtID:机器人ID,一般为0
    *	@param nSlaveID : 从站ID
    *	@param nFunction : 功能码(0x01-读线圈寄存器
                                0x02-读线离散输入寄存器
                                0x03-读保持寄存器
                                0x04-读输入寄存器
                                0x05-写单个线圈寄存器
                                0x06-写单个保持寄存器
                                0x0f-写多个线圈寄存器
                                0x10-写多个保持寄存器)
    *	@param nRegAddr : 寄存器地址
    *	@param nRegCount : 寄存器数量
    *	@param retData[1-n] : 寄存器数据 
    *	@param return: 错误码
    '''

    def HRIF_ReadEndHoldingRegisters(self, boxID, rbtID, nSlaveID, nFunction, nRegAddr, nRegCount, result):
        command = 'ReadHoldingRegisters,'
        command += str(rbtID) + ','
        command += str(nSlaveID) + ','
        command += str(nFunction) + ','
        command += str(nRegAddr) + ','
        command += str(nRegCount) + ',;'
        retData = self.g_clients[boxID].sendAndRecv(command, result)
        return retData

    '''
    *	@index : 
    *	@param brief:等待机器人运动停止
    *	@param return: 错误码
    '''

    def waitMovementDone(self, boxID, rbtID, result):
        while True:
            command = 'ReadCurFSM,'
            command += str(rbtID) + ','
            command += ';'
            retData = self.g_clients[boxID].sendAndRecv(command, result)
            if retData != 0:
                return retData
            if int(result[0]) == 25:
                time.sleep(0.1)
                continue
            else:
                break
        return 0

    # sendVarValue
    # No output

    def cdsSetIO(self, boxID, nEndDOMask, nEndDOVal, nBoxDOMask, nBoxDOVal, nBoxCOMask, nBoxCOVal, nBoxAOCH0_Mask,
                 nBoxAOCH0_Mode, nBoxAOCH1_Mask, nBoxAOCH1_Mode, dbBoxAOCH0_Val, dbBoxAOCH1_Val, result):
        command = 'cdsSetIO,'
        command += str(nEndDOMask) + ','
        command += str(nEndDOVal) + ','
        command += str(nBoxDOMask) + ','
        command += str(nBoxDOVal) + ','
        command += str(nBoxCOMask) + ','
        command += str(nBoxCOVal) + ','
        command += str(nBoxAOCH0_Mask) + ','
        command += str(nBoxAOCH0_Mode) + ','
        command += str(nBoxAOCH1_Mask) + ','
        command += str(nBoxAOCH1_Mode) + ','
        command += str(dbBoxAOCH0_Val) + ','
        command += str(dbBoxAOCH1_Val) + ','
        command += ';'
        return self.g_clients[boxID].sendAndRecv(command, result)


def ReadFloat(*args, reverse=False):
    for n, m in args:
        n, m = '%04x' % n, '%04x' % m
    if reverse:
        v = n + m
    else:
        v = m + n
    y_bytes = bytes.fromhex(v)
    y = struct.unpack('!f', y_bytes)[0]
    y = round(y, 6)
    return y


def WriteFloat(value, reverse=False):
    print(WriteFloat)
    y_bytes = struct.pack('!f', value)
    print(y_bytes)
    y_hex = ''.join(['%02x' % i for i in y_bytes])
    print(y_hex)
    n, m = y_hex[:-4], y_hex[-4:]
    n, m = int(n, 16), int(m, 16)
    if reverse:
        v = [n, m]
    else:
        v = [m, n]
    return v


def ReadDint(*args, reverse=False, result):
    for n, m in args:
        n, m = '%04x' % n, '%04x' % m
    if reverse:
        v = n + m
    else:
        v = m + n
    y_bytes = bytes.fromhex(v)
    y = struct.unpack('!i', y_bytes)[0]
    return y


def WriteDint(value, reverse=False):
    y_bytes = struct.pack('!i', value)
    # y_hex = bytes.hex(y_bytes)
    y_hex = ''.join(['%02x' % i for i in y_bytes])
    n, m = y_hex[:-4], y_hex[-4:]
    n, m = int(n, 16), int(m, 16)
    if reverse:
        v = [n, m]
    else:
        v = [m, n]
    return v


if __name__ == '__main__':
    client = CPSClient()
    boxID = 0  # 假设您使用的是第一个电箱
    rbtID = 0  # 假设您使用的是第一个机器人

    # 连接到电箱和控制器
    client.HRIF_Connect(boxID, '192.168.11.7', 10003)
    client.HRIF_Connect2Controller(boxID)

    current_pose = []
    client.HRIF_ReadActPos(boxID, rbtID, current_pose)
    current_pose = [float(num) for num in current_pose]
    target_pose = current_pose[6:12]
    target_joint = current_pose[0:6]
    print("pose = ",target_pose)
    print("joint = ",target_joint)

    # target_pose[3:6] = [180 - 43.27147704196139, 16.100274074694543, 15.116146464420966]
    target_pose[3:6] = [-90, -20, 0]
    target_pose[0:3] = [50,400,800]
    # client.moveJ_robot([83.795, 4.731, 111.311, -1.832, -77.408, -109.852])
    # client.move_robot(target_pose=target_pose)

    # target_pose = np.array([-0.31735*1000, 0.84271*1000, 0.36603, -180, -0.00040317, 33.258])
    #
    # # target_pose[2] = target_pose[2] + 50
