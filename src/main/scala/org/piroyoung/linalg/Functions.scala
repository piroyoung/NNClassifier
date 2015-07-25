package org.piroyoung.linalg

import scala.math._
/**
 * Created by piroyoung on 7/25/15.
 */
object LinalgFunctions {

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoid(v: Seq[Double]) = v.map(sigmoid(_))

}
