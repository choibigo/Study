using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using EFCore.Models;

namespace EFCore
{
	internal class Program
	{
		static void Main(string[] args)
		{
			// 연결할 DB정의
			MyDbContext db = new MyDbContext(); // 연결할 DB의 객체를 만든다.

			// insert
			Empolyee emp = new Empolyee
			{
				Name = "Daewon",
				salary = 1000
			};

			db.Employees.Add(emp);
			// db 객체에 Employee에 emp를 메모리상 추가 한다.


			Empolyee emp2 = new Empolyee
			{
				Name = "Lil Daewon",
				salary = 100000
			};

			db.Employees.Add(emp2);

			db.SaveChanges(); // db값을 실제 데이터 베이스에 저장한다.

			// select
			var emps = db.Employees.Where(p => p.Id >= 1);
			//연결할DB.찾을 Table.Where절 사용 coloum의 id가 1보다 큰경우 

			foreach (var item in emps)
			{
				Console.WriteLine($"{item.Id} , {item.Name}");
			}
		}
	}
}
