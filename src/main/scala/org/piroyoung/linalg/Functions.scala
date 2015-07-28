package org.piroyoung.linalg

import scala.math._
/**
 * Created by piroyoung on 7/25/15.
 */
object Functions {

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoid(v: Seq[Double]): Seq[Double] = v.map(sigmoid)

  def getOneOfK(label: Int, k: Int): Seq[Double] = {
    if(label > k) throw new IndexOutOfBoundsException
    (0 to (k - 1)).map(x => if(x == k) 1.0 else 0.0)
  }

}
