package org.piroyoung.classficate

import org.piroyoung.linalg.{RowVector, DenseMatrix}
import org.piroyoung.linalg.Functions._
/**
 * Created by piroyoung on 7/25/15.
 */
class NNClassfier {

}

class FeedForwardNetwork(structure: Int*) {
  private val s = structure.dropRight(1)
  val sizes = s.drop(1) zip s.dropRight(1)
  val outputSize = structure.last
  val inputSize = structure(0)
  var layers = sizes.map(x => Layer.init(x._1, x._2))
  def forward(input: RowVector): Seq[RowVector] = {
    var in = input.addBias

    for (l <- layers) yield {
      in = l.forward(in).addBias
      in
    }
  }

  def backword(input: RowVector, answer: RowVector, eta: Double = 0.5): Seq[DenseMatrix] = {
    val outs = forward(input).reverse

    var delta = answer.toSeq.map(a => {
      RowVector(outs(0).toSeq.map(out => -1 * (out - a) * out * (1 - out)))
    }).reduce((x, y) => x + y)

//    for (l <- (layers.map(_.weights) zip outs).reverse.drop(1)) yield {
//      l._2
//    }


  }

}

class Layer(w: DenseMatrix) {
  val weights = w

  def forward(input: RowVector, a: Double => Double = sigmoid): RowVector = (weights * input) activateWith a
  def update(e: DenseMatrix): Layer = new Layer(weights + e)

}
object Layer{
  def init(numNodes: Int, inputSize: Int,  seed: Int = 1234): Layer = new Layer(DenseMatrix(numNodes, inputSize + 1))
}
