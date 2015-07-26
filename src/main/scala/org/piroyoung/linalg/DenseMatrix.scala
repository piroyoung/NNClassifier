package org.piroyoung.linalg

import scala.util.Random

/**
 * Created by piroyoung on 7/25/15.
 */


class DenseMatrix(v: Seq[Seq[Double]]) {

  val values: Seq[Seq[Double]] = v
  val shape: (Int, Int) = (v.length, v(0).length)

  val rowIndices = 0 to (shape._1 - 1)
  val colIndices = 0 to (shape._2 - 1)
  lazy val t = new DenseMatrix(this.colIndices.map(j => this.rowIndices.map(i => this(i, j))))

  override def toString: String = values.map(_.mkString("\t")).mkString("\n")
  def apply(i: Int, j:Int): Double = values(i)(j)
  def row(i: Int):Seq[Double] = values(i)
  def col(j: Int):Seq[Double] = this.rowIndices.map(this(_, j))

  def dropLastCol: DenseMatrix = {
    new DenseMatrix(values.map(_.dropRight(1)))
  }

  def *(that: DenseMatrix): DenseMatrix = {
    val v = this.rowIndices.map(i => {
      that.colIndices.map(k => {
        this.colIndices.map(j => this(i, j) * that(j, k)).sum
      })
    })

    new DenseMatrix(v)
  }

  // looking for better way
  def *(that: ColVector): ColVector = {
    val v = this.rowIndices.map(i => {
      that.colIndices.map(k => {
        this.colIndices.map(j => this(i, j) * that(j, k)).sum
      })
    })
    new ColVector(v)
  }

  def +(that: DenseMatrix): DenseMatrix = {
    val v = this.rowIndices.map(i =>{
      this.colIndices.map(j => {
        this(i, j) + that(i, j)
      })
    })

    new DenseMatrix(v)
  }
  def -(that: DenseMatrix): DenseMatrix = {
    val v = this.rowIndices.map(i =>{
      this.colIndices.map(j => {
        this(i, j) - that(i, j)
      })
    })

    new DenseMatrix(v)
  }

}

object DenseMatrix {
  def getGausiaan(n: Int, m: Int): DenseMatrix = {
    val v = for(i <- Range(0, n)) yield {
      for(j <- Range(0, m)) yield {
        Random.nextGaussian()
      }
    }
    new DenseMatrix(v)
  }
}

class ColVector(v: Seq[Seq[Double]]) extends DenseMatrix(v) {

  def apply(i: Int): Double = values(i)(0)
  def activateWith(f: Double => Double) = new ColVector(values.map(_.map(f)))
  def addBias = ColVector(col(0) :+ 1.0)
  def dropBias = ColVector(col(0).dropRight(1))
  def length = shape._1
  def toSeq = this.col(0)
  def toDenseMatrix = new DenseMatrix(values)

  def *:(that: ColVector): ColVector = {
    ColVector(that.rowIndices.map(i => this(i) * that(i)))
  }
}

object ColVector {
  def apply(v: Seq[Double]) = {
    new ColVector(v.map(Seq(_)))
  }
//  def apply(v: Double*) = {
//    new ColVector(v.map(Seq(_)))
//  }
  def getOnes(n: Int): ColVector = ColVector(Seq.fill(n)(1.0))
}


