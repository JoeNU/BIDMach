package BIDMach.allreduce.binder

import BIDMach.allreduce.binder.AllreduceBinder.{DataSink, DataSource}
import BIDMach.models.Model
import BIDMat.{FMat, IMat, Mat}


/**
  * Linearize input model mats, and elastic-average update to the same model.
  *
  * @param model
  * @param alpha
  */
class ElasticAverageBinder(model: Model, alpha: Double) extends AllreduceBinder {

  override lazy val totalDataSize: Int = {
    var ret = 0
    model.modelmats.synchronized {
      for (mat <- model.modelmats) {
        ret += mat.length
      }
    }
    ret
  }

  override def dataSource: DataSource = inputRequest => {

    println(s"--Dumping model data at ${inputRequest.iteration}--")
    val ret: Array[Float] = new Array[Float](totalDataSize)

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    var i = model.modelmats.length - 1

    while (i >= 0) {
      val mat = model.modelmats(i)
      mat.synchronized {
        val contentData = FMat(mat).contents.data
        current -= contentData.length
        System.arraycopy(contentData, 0, ret, current, contentData.length)
      }
      i -= 1
    }

    AllReduceInput(ret)

  }

  private def averageValueOrElse(sum: Float, count: Int): Option[Float] = {
    count match {
      case 0 => Option.empty
      case _ => Some(sum / count)
    }
  }

  override def dataSink: DataSink = reducedOutput => {
    println(s"-- Averaging model of iteration ${reducedOutput.iteration}--")

    val data = reducedOutput.data
    val count = reducedOutput.count

    assert(data.length == totalDataSize, "Reduced output should be the same as as model")

    // backward traversing model mats, assuming forward traversal by the training model
    // using while instead of for loop due to performance
    var current = totalDataSize
    var i = model.modelmats.length - 1

    // trasnfer to fmat so that we can just add average

    while (i >= 0) {
      val mat = model.modelmats(i)
      val content = FMat(mat).data
      val contentLength = mat.length
      var j = 0
      current -= contentLength
      while (j < contentLength) {
        val avgVal = data(current + j) / count(current + j)
        val diff = avgVal - content(j)
        mat.update(j, content(j) + alpha * diff )
        j += 1
      }
      i -= 1
    }
    assert(current == 0, "current should be zero after iteration")

  }

}
