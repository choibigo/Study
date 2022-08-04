from concurrent import futures

import logging

import grpc
import greet_pb2
import greet_pb2_grpc
import arithmetic_pb2
import arithmetic_pb2_grpc



def run1():
    with grpc.insecure_channel('localhost:5140') as channel:
    	# stub을 생성해줍니다.
        stub = greet_pb2_grpc.GreeterStub(channel)
        
        # 요청을 보내고 결과를 받는데, 서버에서 지정한 메서드에 요청시 사용할 proto 메시지 형식으로 요청을 전송합니다.
        response = stub.SayHello(greet_pb2.HelloRequest(name='you'))
        print("Greeter client received: " + response.message)

def run2():
    with grpc.insecure_channel('localhost:5140') as channel:
        stub = arithmetic_pb2_grpc.ArithmeticServiceStub(channel)

        response = stub.Add(arithmetic_pb2.RequestMessage(first = 1, second = 10))
        print(f"Add Result : {response.result}")

if __name__ == '__main__':
    logging.basicConfig()
    run1()
    run2()