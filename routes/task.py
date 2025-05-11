from fastapi import FastAPI, HTTPException
from jet.logger import logger
from jet.llm.mlx.server.task_manager import TaskManager

app = FastAPI()
task_manager = TaskManager()


@app.get("/tasks")
async def get_tasks():
    try:
        return {"tasks": task_manager.get_all_tasks()}
    except Exception as e:
        logger.error(f"Failed to retrieve tasks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve tasks: {str(e)}")


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to retrieve task {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve task: {str(e)}")


@app.delete("/tasks")
async def clear_tasks():
    try:
        task_manager.repository.reset_schema()
        task_manager.tasks.clear()
        return {"message": "All tasks cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear tasks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to clear tasks: {str(e)}")


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        with task_manager.repository.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM prompts WHERE task_id = %s", (task_id,))
                cur.execute("DELETE FROM tasks WHERE task_id = %s", (task_id,))
                conn.commit()
        with task_manager.lock:
            if task_id in task_manager.tasks:
                del task_manager.tasks[task_id]
        return {"message": f"Task {task_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to delete task {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to delete task {task_id}: {str(e)}")
