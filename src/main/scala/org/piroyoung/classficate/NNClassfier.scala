package org.piroyoung.classficate

import java.io.{BufferedWriter, FileWriter}

import org.piroyoung.linalg.Functions._
import org.piroyoung.linalg.{ColVector, DenseMatrix}

import scala.io.Source

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
  private var act: Double => Double = dropSigmoid

  def setEta(e: Double): FeedForwardNetwork = {
    eta = e
    this
  }

  def setActivator(f: Double => Double): FeedForwardNetwork = {
    act = f
    this
  }

  override def toString(): String = layers.map(_.weights).map(_.toString).mkString("\n---\n")

  def forward(input: ColVector, a: Double => Double = sigmoid): Seq[ColVector] = {
    var in = input.addBias
    for (l <- layers) yield {
      in = l.forward(in, a).addBias
      in
    }
  }

  def backward(input: ColVector, answer: ColVector, a: Double => Double = sigmoid): Seq[DenseMatrix] = {
    val outs = forward(input, a)
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

  def update(input: ColVector, answer: ColVector, a: Double => Double = sigmoid): Unit = {
    val grads = backward(input, answer, a)
    (layers zip grads).foreach(l => l._1.update(l._2 * eta))

    val v = predict(input).toSeq
    println(v.indexOf(v.max))
    println(v.toString())
  }

  def fit(data: Seq[(ColVector, Double)], k: Int, iter: Int): FeedForwardNetwork = {
    for (i <- Range(0, iter); d <- data) {
      print(d._2.toString + "::")
      update(d._1, ColVector.getOneOfK(d._2, k), act)
    }
    new FeedForwardNetwork(layers)
  }

  def predict(input: ColVector): ColVector = {
    forward(input, sigmoid).last.dropBias
  }

  def predictLabel(input: ColVector): Double = {
    val v = predict(input).toSeq
    v.indexOf(v.max)
  }

  def saveAsTextFile(fileName: String): Unit = {
    val bw = new BufferedWriter(new FileWriter(fileName))
    for (line <- this.toString().split("\n")) {
      bw.write(line + "\n")
    }

    bw.close()
  }

}

object FeedForwardNetwork {
  def apply(structure: Int*): FeedForwardNetwork = {
    val s = structure
    val sizes = s.drop(1) zip s.dropRight(1)
    val layers = sizes.map(x => Layer.init(x._1, x._2))
    new FeedForwardNetwork(layers)
  }

  def load(fileName: String): FeedForwardNetwork = {
    val l = Source.fromFile(fileName).getLines()
      .mkString("\n")
      .split("\n---\n")
      .map(
        _.split("\n").toSeq.map(
          _.split("\t").toSeq.map(_.toDouble)
        )
      )
      .map(m => new Layer(new DenseMatrix(m)))

    new FeedForwardNetwork(l)
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

}

object Layer {
  def init(numNodes: Int, inputSize: Int, seed: Int = 1234): Layer = {
    new Layer(DenseMatrix.getGausiaan(numNodes, inputSize + 1))
  }
}
