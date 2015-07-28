package org.piroyoung.classficate

import org.piroyoung.linalg.Functions._
import org.piroyoung.linalg.{ColVector, DenseMatrix}

/**
 * Created by piroyoung on 7/25/15.
 */

class FeedForwardNetwork(l: Seq[Layer]) extends Serializable {
  //  private val s = structure
  //  val sizes = s.drop(1) zip s.dropRight(1)
  //  val outputSize = structure.last
  //  val inputSize = structure(0)
  var layers = l
  private var eta: Double = 1

  def setEta(e: Double): Unit = eta = e

  override def toString(): String = layers.map(_.toString()).mkString("\n---\n")

  def forward(input: ColVector): Seq[ColVector] = {
    var in = input.addBias
    for (l <- layers) yield {
      in = l.forward(in).addBias
      in
    }
  }

  def backward(input: ColVector, answer: ColVector): Seq[DenseMatrix] = {
    val outs = forward(input)
    val inputs = input.addBias +: outs.dropRight(1)
    val o = outs.last
    val lastDelta = ColVector(answer.rowIndices.map(i => {
      -1 * (answer(i) - o(i)) * o(i) * (1 - o(i))
    }))

    val layerInputs = layers zip inputs
    var d = lastDelta
    val dd = for (l <- layerInputs.reverse) yield {
      d = l._1.backward(l._2, d)
      d
    }

    val deltas = (lastDelta +: dd).dropRight(1).reverse

    deltas.indices.map(j => {
      deltas(j) * inputs(j).t
    })
  }

  def fit(input: ColVector, answer: ColVector): FeedForwardNetwork = {
    val grads = backward(input, answer)
    (layers zip grads).foreach(l => l._1.update(l._2 * eta))
    this
  }

  def fit(input: ColVector, answer: ColVector, iter: Int): FeedForwardNetwork = {
    for(i <- 0 to iter) {
      fit(input, answer)
    }
    this
  }

  def predict(input: ColVector): ColVector = {
    forward(input).last.dropBias
  }

  def combine(that: FeedForwardNetwork): FeedForwardNetwork = {
    new FeedForwardNetwork((layers zip that.layers).map(x => x._1 combine (x._2)))
  }
}

object FeedForwardNetwork {
  def apply(structure: Int*): FeedForwardNetwork = {
    val s = structure
    val sizes = s.drop(1) zip s.dropRight(1)
    val layers = sizes.map(x => Layer.init(x._1, x._2))
    new FeedForwardNetwork(layers)
  }
}

class Layer(w: DenseMatrix) {
  var weights = w

  override def toString(): String = weights.toString

  def update(e: DenseMatrix): Unit = {
    weights = weights - e
  }

  def forward(input: ColVector, a: Double => Double = sigmoid): ColVector = (weights * input) activateWith a

  // returns previous deltas
  // *: culcs element-wise production
  def backward(input: ColVector, thisDelta: ColVector): ColVector = {
    (weights.dropLastCol.t * thisDelta) *: ColVector(input.dropBias.toSeq.map(y => y * (1 - y)))
  }

  def combine(that: Layer) = {
    new Layer((weights + that.weights) / 2)
  }

}

object Layer {
  def init(numNodes: Int, inputSize: Int, seed: Int = 1234): Layer = {
    new Layer(DenseMatrix.getGausiaan(numNodes, inputSize + 1))
  }
}
