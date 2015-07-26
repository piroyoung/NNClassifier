package org.piroyoung.classficate

import org.piroyoung.linalg.{RowVector, DenseMatrix}
import org.piroyoung.linalg.Functions._
/**
 * Created by piroyoung on 7/25/15.
 */

class FeedForwardNetwork(structure: Int*) {
  private val s = structure
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

//  def backword(input: RowVector, answer: RowVector, eta: Double = 0.5): Seq[DenseMatrix] = {
//    val outs = forward(input)
//
//    val o = outs.last
//    val delta = answer.colIndices.map(i => {
//      -1 * (answer(i) - o(i)) * o(i) * (1 - o(i))
//    })
//
//  }


}

class Layer(w: DenseMatrix) {
  var weights = w

  def forward(input: RowVector, a: Double => Double = sigmoid): RowVector = (weights * input) activateWith a
  def update(e: DenseMatrix): Layer = new Layer(weights + e)

  def backward(output: RowVector, nextDelta: RowVector): DenseMatrix = {
    //FIXME
    new DenseMatrix(Seq(Seq()))
  }

}
object Layer{
  def init(numNodes: Int, inputSize: Int,  seed: Int = 1234): Layer = new Layer(DenseMatrix(numNodes, inputSize + 1))
}
