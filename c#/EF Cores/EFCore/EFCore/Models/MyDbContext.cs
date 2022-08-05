using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;

namespace EFCore.Models
{
	// DbContext가 데이터베이스의 골격을 만들어 준다.
	public class MyDbContext : DbContext
	{
		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		// 이데이터를 설정하는 메소드를 오버라이드 한다.
		{
			optionsBuilder.UseSqlServer("Data Source=(local);Initial Catalog=TestDb;Integrated Security=True; TrustServerCertificate=True");
		}

		public DbSet<Empolyee> Employees { get; set; } // Employees 속성을 통해 이 데이터 베이스의 테이블 데이터를 엑세스 할 수 있다.
	}

	// 데이터 베이스에 실제 테이블을 정의 한다.
	// 이 데이터 베이스는 Empolyee테이블을 갖는다.
}
