using Grpc.Core;
using Grpc.Server;

namespace Grpc.Server.Services
{
	public class GreeterService : Greeter.GreeterBase
	{
		private readonly ILogger<GreeterService> _logger;
		public GreeterService(ILogger<GreeterService> logger)
		{
			_logger = logger;
		}

		public override Task<HelloReply> SayHello(HelloRequest request, ServerCallContext context)
		{
			Console.WriteLine("Server Call Success");
			
			return Task.FromResult(new HelloReply
			{
				Message = "Hello My Name Is" + request.Name
			});
		}
	}
}