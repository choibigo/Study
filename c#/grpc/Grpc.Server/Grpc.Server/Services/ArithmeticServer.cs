// ========= 서버 구현 =========

using ArithmeticNamespace;
using Grpc.Core;


/*
proto file의 
option csharp_namespace = "ArithmeticNamespace";
에서 정의한 이름을 사용한다. 
*/



namespace Grpc.Server
{
	public class ArithmeticServer : ArithmeticService.ArithmeticServiceBase
		// MyNameService.MyNameServeiceBase를 상속 받는다.
		// 자동으로 XXXBase 클래스가 만들어 진다.
	{
		// ArithmeticService내에서 사용하는 함수 실제 구현
		public override Task<ResponseMessage> Add(RequestMessage request, ServerCallContext context)
		/*
			반환 값은 Proto에서 정의된 ResponseMessage 을 사용하고, Task Type으로 반환 한다. 
			RequestMessage Type의 입력이다.
		 */
		{
			var r = new ResponseMessage(); // 반환값 선언
			r.Result = request.First + request.Second; // 실제 구현

			return Task.FromResult(r); // Task 형태로 반환
		}

		public override Task<ResponseMessage> Substract(RequestMessage request, ServerCallContext context)
		{
			var r = new ResponseMessage();
			r.Result = request.First - request.Second;

			return Task.FromResult(r);
		}
	}
}
