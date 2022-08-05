using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace EFCore.Models
{
	//[Table("EMP")] // C# 에서는 Empolyee라고 하고 SQL 에서는 EMP라 한다. 를 명시
	public class Empolyee
	{
		//[Key] // ID 가 Primary Key임을 명시 
		public int Id { get; set; } // id 이름으로 만들면 자동으로 Primary Key가 된다.
		public string Name { get; set; }
		public decimal salary { get; set; }
	}
	// 테이블을 구조(메타데이터)를 정의 한다.
	// 컬럼을 정의하고 컬럼명을 정의 한다.
}
