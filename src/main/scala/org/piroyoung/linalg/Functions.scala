package org.piroyoung.linalg

import scala.math._
import scala.util.Random

/**
 * Created by piroyoung on 7/25/15.
 */
object Functions {
  val r = new Random(1234)

  def sigmoid(x: Double) = 1 / (1 + exp(-x))

  def sigmoid(v: Seq[Double]): Seq[Double] = v.map(sigmoid)

  def dropSigmoid(x: Double) = if(r.nextBoolean) sigmoid(x) else 0.0

  def dropSigmoid(v: Seq[Double]): Seq[Double] = v.map(dropSigmoid)

  def rectifier(x: Double) = max(0.0, x)

  def rectifier(v: Seq[Double]): Seq[Double] = v.map(rectifier)

  def getOneOfK(label: Int, k: Int): ColVector = {
    if (label > k) throw new IndexOutOfBoundsException
    ColVector((0 to (k - 1)).map(x => if (x == label) 1.0 else 0.0))
  }

}
